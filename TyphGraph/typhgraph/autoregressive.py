# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A Predictor wrapping a one-step Predictor to make autoregressive predictions.
"""

from typing import Optional, cast

from absl import logging
from graphcast import predictor_base
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import xarray


def _unflatten_and_expand_time(flat_variables, tree_def, time_coords):
  variables = jax.tree_util.tree_unflatten(tree_def, flat_variables)
  return variables.expand_dims(time=time_coords, axis=0)


def _get_flat_arrays_and_single_timestep_treedef(variables):
  flat_arrays = jax.tree_util.tree_leaves(variables.transpose('time', ...))
  _, treedef = jax.tree_util.tree_flatten(variables.isel(time=0, drop=True))
  return flat_arrays, treedef


class Predictor(predictor_base.Predictor):
  """Wraps a one-step Predictor to make multi-step predictions autoregressively.

  The wrapped Predictor will be used to predict a single timestep conditional
  on the inputs passed to the outer Predictor. Its predictions are then
  passed back in as inputs at the next timestep, for as many timesteps as are
  requested in the targets_template. (When multiple timesteps of input are
  used, a rolling window of inputs is maintained with new predictions
  concatenated onto the end).

  You may ask for additional variables to be predicted as targets which aren't
  used as inputs. These will be predicted as output variables only and not fed
  back in autoregressively. All target variables must be time-dependent however.

  You may also specify static (non-time-dependent) inputs which will be passed
  in at each timestep but are not predicted.

  At present, any time-dependent inputs must also be present as targets so they
  can be passed in autoregressively.

  The loss of the wrapped one-step Predictor is averaged over all timesteps to
  give a loss for the autoregressive Predictor.
  """

  def __init__(
      self,
      predictor: predictor_base.Predictor,
      noise_level: Optional[float] = None,
      gradient_checkpointing: bool = False,
      ):
    """Initializes an autoregressive predictor wrapper.

    Args:
      predictor: A predictor to wrap in an auto-regressive way.
      noise_level: Optional value that multiplies the standard normal noise
        added to the time-dependent variables of the predictor inputs. In
        particular, no noise is added to the predictions that are fed back
        auto-regressively. Defaults to not adding noise.
      gradient_checkpointing: If True, gradient checkpointing will be
        used at each step of the computation to save on memory. Roughtly this
        should make the backwards pass two times more expensive, and the time
        per step counting the forward pass, should only increase by about 50%.
        Note this parameter will be ignored with a warning if the scan sequence
        length is 1.
    """
    self._predictor = predictor
    self._noise_level = noise_level
    self._gradient_checkpointing = gradient_checkpointing

  def _get_and_validate_constant_inputs(self, inputs, targets, forcings):
    constant_inputs = inputs.drop_vars(targets.keys(), errors='ignore')
    constant_inputs = constant_inputs.drop_vars(
        forcings.keys(), errors='ignore')
    for name, var in constant_inputs.items():
      if 'time' in var.dims:
        raise ValueError(
            f'Time-dependent input variable {name} must either be a forcing '
            'variable, or a target variable to allow for auto-regressive '
            'feedback.')
    return constant_inputs

  def _validate_targets_and_forcings(self, targets, forcings):
    for name, var in targets.items():
      if 'time' not in var.dims:
        raise ValueError(f'Target variable {name} must be time-dependent.')

    for name, var in forcings.items():
      if 'time' not in var.dims:
        raise ValueError(f'Forcing variable {name} must be time-dependent.')

    overlap = forcings.keys() & targets.keys()
    if overlap:
      raise ValueError('The following were specified as both targets and '
                       f'forcings, which isn\'t allowed: {overlap}')

  def _update_inputs(self, inputs, next_frame):
    num_inputs = inputs.dims['time']

    predicted_or_forced_inputs = next_frame[list(inputs.keys())]

    # Combining datasets with inputs and target time stamps aligns them.
    # Only keep the num_inputs trailing frames for use as next inputs.
    return (xarray.concat([inputs, predicted_or_forced_inputs], dim='time')
            .tail(time=num_inputs)
            # Update the time coordinate to reset the lead times for
            # next AR iteration.
            .assign_coords(time=inputs.coords['time']))

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               **kwargs) -> xarray.Dataset:
    """Calls the Predictor.

    Args:
      inputs: input variable used to make predictions. Inputs can include both
        time-dependent and time independent variables. Any time-dependent
        input variables must also be present in the targets_template or the
        forcings.
      targets_template: A target template containing informations about which
        variables should be predicted and the time alignment of the predictions.
        All target variables must be time-dependent.
        The number of time frames is used to set the number of unroll of the AR
        predictor (e.g. multiple unroll of the inner predictor for one time step
        in the targets is not supported yet).
      forcings: Variables that will be fed to the model. The variables
        should not overlap with the target ones. The time coordinates of the
        forcing variables should match the target ones.
        Forcing variables which are also present in the inputs, will be used to
        supply ground-truth values for those inputs when they are passed to the
        underlying predictor at timesteps beyond the first timestep.
      **kwargs: Additional arguments passed along to the inner Predictor.

    Returns:
      predictions: the model predictions matching the target template.

    Raise:
      ValueError: if the time coordinates of the inputs and targets are not
        different by a constant time step.
    """

    constant_inputs = self._get_and_validate_constant_inputs(
        inputs, targets_template, forcings)
    self._validate_targets_and_forcings(targets_template, forcings)

    # After the above checks, the remaining inputs must be time-dependent:
    inputs = inputs.drop_vars(constant_inputs.keys())

    # A predictions template only including the next time to predict.
    target_template = targets_template.isel(time=[0])

    flat_forcings, forcings_treedef = (
        _get_flat_arrays_and_single_timestep_treedef(forcings))
    scan_variables = flat_forcings

    def one_step_prediction(inputs, scan_variables):

      flat_forcings = scan_variables
      forcings = _unflatten_and_expand_time(flat_forcings, forcings_treedef,
                                            target_template.coords['time'])

      # Add constant inputs:
      all_inputs = xarray.merge([constant_inputs, inputs])
      predictions: xarray.Dataset = self._predictor(
          all_inputs, target_template,
          forcings=forcings,
          **kwargs)

      next_frame = xarray.merge([predictions, forcings])
      next_inputs = self._update_inputs(inputs, next_frame)

      # Drop the length-1 time dimension, since scan will concat all the outputs
      # for different times along a new leading time dimension:
      predictions = predictions.squeeze('time', drop=True)
      # We return the prediction flattened into plain jax arrays, because the
      # extra leading dimension added by scan prevents the tree_util
      # registrations in xarray_jax from unflattening them back into an
      # xarray.Dataset automatically:
      flat_pred = jax.tree_util.tree_leaves(predictions)
      return next_inputs, flat_pred

    if self._gradient_checkpointing:
      scan_length = targets_template.dims['time']
      if scan_length <= 1:
        logging.warning(
            'Skipping gradient checkpointing for sequence length of 1')
      else:
        # Just in case we take gradients (e.g. for control), although
        # in most cases this will just be for a forward pass.
        one_step_prediction = hk.remat(one_step_prediction)

    # Loop (without unroll) with hk states in cell (jax.lax.scan won't do).
    _, flat_preds = hk.scan(one_step_prediction, inputs, scan_variables)

    # The result of scan will have an extra leading axis on all arrays,
    # corresponding to the target times in this case. We need to be prepared for
    # it when unflattening the arrays back into a Dataset:
    scan_result_template = (
        target_template.squeeze('time', drop=True)
        .expand_dims(time=targets_template.coords['time'], axis=0))
    _, scan_result_treedef = jax.tree_util.tree_flatten(scan_result_template)
    predictions = jax.tree_util.tree_unflatten(scan_result_treedef, flat_preds)
    return predictions

  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: xarray.Dataset,
           **kwargs
           ) -> predictor_base.LossAndDiagnostics:
    """The mean of the per-timestep losses of the underlying predictor."""
    if targets.sizes['time'] == 1:
      # If there is only a single target timestep then we don't need any
      # autoregressive feedback and can delegate the loss directly to the
      # underlying single-step predictor. This means the underlying predictor
      # doesn't need to implement .loss_and_predictions.
      return self._predictor.loss(inputs, targets, forcings, **kwargs)

    constant_inputs = self._get_and_validate_constant_inputs(
        inputs, targets, forcings)
    self._validate_targets_and_forcings(targets, forcings)
    # After the above checks, the remaining inputs must be time-dependent:
    inputs = inputs.drop_vars(constant_inputs.keys())

    if self._noise_level:
      def add_noise(x):
        return x + self._noise_level * jax.random.normal(
            hk.next_rng_key(), shape=x.shape)
      # Add noise to time-dependent variables of the inputs.
      inputs = jax.tree.map(add_noise, inputs)

    # The per-timestep targets passed by scan to one_step_loss below will have
    # no leading time axis. We need a treedef without the time axis to use
    # inside one_step_loss to unflatten it back into a dataset:
    flat_targets, target_treedef = _get_flat_arrays_and_single_timestep_treedef(
        targets)
    scan_variables = flat_targets

    flat_forcings, forcings_treedef = (
        _get_flat_arrays_and_single_timestep_treedef(forcings))
    scan_variables = (flat_targets, flat_forcings)

    def one_step_loss(inputs, scan_variables):
      flat_target, flat_forcings = scan_variables
      forcings = _unflatten_and_expand_time(flat_forcings, forcings_treedef,
                                            targets.coords['time'][:1])

      target = _unflatten_and_expand_time(flat_target, target_treedef,
                                          targets.coords['time'][:1])

      # Add constant inputs:
      all_inputs = xarray.merge([constant_inputs, inputs])

      (loss, diagnostics), predictions = self._predictor.loss_and_predictions(
          all_inputs,
          target,
          forcings=forcings,
          **kwargs)

      # Unwrap to jax arrays shape (batch,):
      loss, diagnostics = xarray_tree.map_structure(
          xarray_jax.unwrap_data, (loss, diagnostics))

      predictions = cast(xarray.Dataset, predictions)  # Keeps pytype happy.
      next_frame = xarray.merge([predictions, forcings])
      next_inputs = self._update_inputs(inputs, next_frame)

      return next_inputs, (loss, diagnostics)

    if self._gradient_checkpointing:
      scan_length = targets.dims['time']
      if scan_length <= 1:
        logging.warning(
            'Skipping gradient checkpointing for sequence length of 1')
      else:
        one_step_loss = hk.remat(one_step_loss)

    # We can pass inputs (the initial state of the loop) in directly as a
    # Dataset because the shape we pass in to scan is the same as the shape scan
    # passes to the inner function. But, for scan_variables, we must flatten the
    # targets (and unflatten them inside the inner function) because they are
    # passed to the inner function per-timestep without the original time axis.
    # The same apply to the optional forcing.
    _, (per_timestep_losses, per_timestep_diagnostics) = hk.scan(
        one_step_loss, inputs, scan_variables)

    # Re-wrap loss and diagnostics as DataArray and average them over time:
    (loss, diagnostics) = jax.tree_util.tree_map(
        lambda x: xarray_jax.DataArray(x, dims=('time', 'batch')).mean(  # pylint: disable=g-long-lambda
            'time', skipna=False),
        (per_timestep_losses, per_timestep_diagnostics))

    return loss, diagnostics


# 좋아. 이 autoregressive.py 파일은 GraphCast에서 핵심적인 역할을 하는 Autoregressive Predictor를 정의하고 있어. 쉽게 말하면, 한 스텝씩 예측하는 기존 모델을 "여러 스텝" 예측하는 모델로 래핑(wrapping)해서, 시간이 흐름에 따라 예측 결과를 다시 입력에 넣는 방식으로 작동하도록 만들어 주는 클래스야.

# ✅ 전체 구조 요약
# plaintext
# 복사
# 편집
# 하나의 Predictor (1-step) → Autoregressive Predictor (multi-step)
# 즉, GraphCast는 기본적으로 "6시간 후" 한 스텝만 예측할 수 있는데, 이걸 여러 번 반복해서 6h → 12h → 18h → 24h 예측으로 확장해주는 거야. 이 파일이 바로 그 기능을 해.

# 🔧 주요 구성 요소 요약
# 1. Predictor 클래스 정의
# 이 클래스는 predictor_base.Predictor를 상속.

# 내부에 진짜 예측을 수행하는 1-step predictor (self._predictor)를 가지고 있음.

# 목적: 이 1-step predictor를 여러 번 호출하면서 autoregressive하게 예측하는 wrapper를 만드는 것.

# 2. 주요 함수 설명
# ✅ __call__
# python
# 복사
# 편집
# def __call__(self, inputs, targets_template, forcings, **kwargs)
# inputs: 예측에 필요한 입력 (과거 데이터들 포함)

# targets_template: 몇 시간 뒤까지 예측할지 정의

# forcings: 기후 외부 영향 변수들 (예: 태양복사, topography 등)

# autoregressive하게 여러 스텝 예측을 수행함.

# 내부적으로 hk.scan을 써서 반복적으로 예측하고 결과를 이어붙임.

# ✅ loss
# python
# 복사
# 편집
# def loss(self, inputs, targets, forcings, **kwargs)
# Autoregressive 방식으로 여러 스텝에 대해 예측한 다음, 각 스텝의 loss를 계산해서 평균냄.

# loss를 통해 전체 rollout 과정이 얼마나 잘 맞는지 학습할 수 있음.

# 🔁 Autoregressive Flow
# 초기 입력 + constant input + forcings로 1-step 예측 수행

# 예측 결과를 다시 입력에 넣음 (_update_inputs)

# 위 과정을 rollout할 길이만큼 반복 (hk.scan 이용)

# 예측 결과를 전부 이어붙여 최종 예측 결과를 반환

# 🔍 함수별 디테일 포인트

# 함수 이름	기능
# _get_and_validate_constant_inputs	시간축 없는 입력값만 골라냄 (static input)
# _validate_targets_and_forcings	target과 forcing이 겹치지 않게 체크
# _update_inputs	예측 결과를 다시 inputs로 반영함
# _get_flat_arrays_and_single_timestep_treedef	flatten + treedef 저장 (JAX-friendly 구조)
# _unflatten_and_expand_time	flatten된 값 → 다시 Dataset으로 복원
# 📌 핵심 포인트 요약
# graphcast.py에서 쓰이는 예측기는 실제론 이 Predictor 클래스야.

# rollout 기반 예측, 즉 autoregressive한 multi-step prediction의 중심축이 이 파일.

# 이걸 수정하면 rollout length 늘리기, scheduled sampling, rollout loss, physics loss 등 추가가 가능해.

# 필요하면 여기서 rollout loss나 physics constraint를 넣을 수 있는 지점도 짚어줄게.
# 다음으로 그걸 볼까?

# 여기는 학습시키는 부분은 아닌거야?


# ChatGPT의 말:
# 맞아, 아주 정확하게 봤어.

# 이 autoregressive.py 파일은 "학습 자체를 수행하는 부분은 아니야."
# 대신 학습에 필요한 forward pass, loss 계산 구조를 정의하는 "모듈"이야. 즉, **"어떻게 예측할지 / loss를 계산할지"**에 대한 로직만 포함되어 있어.

# ✅ 정확히는 어떤 역할을 하냐면:
# 🔹 1. __call__() 함수
# → forward pass 정의
# 즉, 모델이 주어진 입력을 가지고 어떻게 autoregressive하게 예측을 해나가는지를 정의

# 🔹 2. loss() 함수
# → loss 계산 방식 정의
# 여러 time step에 대해 autoregressive하게 예측한 뒤, 각 step의 loss를 구하고 평균을 내서 최종 loss로 반환

# 따라서 이 파일은 모델을 학습시키는 루프(for epoch in ...) 같은 건 전혀 없음.

# 🧠 그럼 학습은 어디서 하냐?
# 학습은 보통 다음과 같은 코드에서 이루어져:

# python
# 복사
# 편집
# # 예시
# loss, diagnostics = loss_fn_jitted(
#     rng=rng_key,
#     inputs=train_inputs,
#     targets=train_targets,
#     forcings=train_forcings
# )

# grads = jax.grad(loss_fn)(...)
# optimizer.apply(grads)
# 이런 코드는 GraphCast 프로젝트에서는 보통 train.py 또는 실험용 notebook 등에서 수행하고,
# 그때 이 Predictor.loss()가 호출되어 loss를 구하게 되는 거야.

# 📌 정리하면

# 역할	어디서?	설명
# 예측 (forward)	__call__()	autoregressive 방식으로 예측
# loss 계산	loss()	각 step의 loss를 평균
# 학습 loop	(다른 파일)	이 Predictor를 사용해 실제 학습 수행
# 학습 시 쓰는 loss_fn, grads_fn, optimizer, train loop 쪽이 궁금하면 그 쪽 파일을 같이 보면 돼. 볼래?






# 좋은 질문이야. `autoregressive.py` 안의 `Predictor` 클래스와 `rollout.py`에 있는 `rollout()` 함수는 겉보기엔 비슷해 보여도 **역할과 목적이 확실히 달라.**

# ---

# ## 🎯 TL;DR 차이 요약

# | 구분 | `autoregressive.py`의 `Predictor` | `rollout.py`의 `rollout()` 함수 |
# |------|------------------------------|-----------------------------|
# | 📦 **목적** | 학습 중에 multi-step prediction & loss 계산 수행 | **학습된 모델**로 inference할 때 예측 생성 |
# | 🧠 **사용 시점** | 학습 중 (forward, loss 계산용) | 추론(inference) 또는 평가(evaluation) 시점 |
# | 🔁 **사용 방식** | 내부에서 `hk.scan()` 등으로 반복 예측 | 훈련된 모델을 불러와서 전체 rollout 생성 |
# | 🧩 **Gradient 사용** | 있음 (JAX grad 가능) | 없음 (보통 no grad, 평가 전용) |

# ---

# ## 🧩 구조상 차이

# ### 🔹 `autoregressive.py`의 `Predictor` 클래스

# - 학습할 때 사용되는 **JAX haiku 모듈 형태**
# - `__call__()` : 예측 수행 (rollout 방식)
# - `loss()` : 여러 스텝의 예측 loss 평균
# - `hk.remat()` / `hk.scan()` 사용 → gradient flow 관리
# - 파라미터 업데이트와 연계됨 (즉, gradient 계산 목적)

# ```python
# (loss, diagnostics), predictions = self._predictor.loss_and_predictions(...)
# ```

# → 결국 이건 **학습 중 백프로 들어감**

# ---

# ### 🔹 `rollout.py`의 `rollout()` 함수

# - 학습 후 모델을 실제로 **예측에 활용하는 함수**
# - 예: "2025-04-17 00:00 기준으로 앞으로 24시간 날씨 예측해줘"
# - 파라미터는 고정 (학습된 모델 사용)
# - gradient 계산 없음

# ```python
# # rollout.py 예시
# output = rollout(predict_fn, input_state, rollout_times, forcings)
# ```

# → **이건 inference 용도로만 사용돼**

# ---

# ## 📌 비유로 정리

# - `autoregressive.py`: 모델 학습 중 "한 스텝 예측 → 다음 입력에 반영" 구조를 만들어 주는 **코어 학습 로직**
# - `rollout.py`: 학습된 모델을 가지고 실제로 예측을 펼쳐보는 **실행 도우미 함수**

# ---

# ## 🔧 내부적으로 비슷한 점?

# - 둘 다 **autoregressive한 구조**를 따름
# - 둘 다 예측 결과를 다음 입력으로 사용
# - 다만 `rollout()`은 학습에 필요한 JAX loss 계산이나 gradient를 쓰지 않음

# ---

# ## ✅ 결론

# - 학습할 땐 `autoregressive.Predictor`를 사용해서 loss까지 모두 반영하며 예측
# - 학습이 끝난 후 평가하거나 실제 날씨를 예측할 땐 `rollout.py`의 `rollout()` 사용

# ---

# `rollout.py` 코드도 같이 분석해볼까? 아니면 두 코드를 비교해서 어떤 걸 수정해야 fine-tuning에 rollout loss 넣을 수 있을지 짚어줄까?