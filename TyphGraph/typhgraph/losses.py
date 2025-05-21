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
"""Loss functions (and terms for use in loss functions) used for weather."""

from typing import Mapping

import xarray_tree
import numpy as np
from typing_extensions import Protocol
import xarray

LossAndDiagnostics = tuple[xarray.DataArray, xarray.Dataset]

class LossFunction(Protocol):
  """A loss function.

  This is a protocol so it's fine to use a plain function which 'quacks like'
  this. This is just to document the interface.
  """

  def __call__(self,
               predictions: xarray.Dataset,
               targets: xarray.Dataset,
               **optional_kwargs) -> LossAndDiagnostics:
    """Computes a loss function.

    Args:
      predictions: Dataset of predictions.
      targets: Dataset of targets.
      **optional_kwargs: Implementations may support extra optional kwargs.

    Returns:
      loss: A DataArray with dimensions ('batch',) containing losses for each
        element of the batch. These will be averaged to give the final
        loss, locally and across replicas.
      diagnostics: Mapping of additional quantities to log by name alongside the
        loss. These will will typically correspond to terms in the loss. They
        should also have dimensions ('batch',) and will be averaged over the
        batch before logging.
    """


def weighted_mse_per_level(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    per_variable_weights: Mapping[str, float],
) -> LossAndDiagnostics:
  """Latitude- and pressure-level-weighted MSE loss."""
  def loss(prediction, target):
    loss = (prediction - target)**2
    loss *= normalized_latitude_weights(target).astype(loss.dtype)
    if 'level' in target.dims:
      loss *= normalized_level_weights(target).astype(loss.dtype)
    return _mean_preserving_batch(loss)

  losses = xarray_tree.map_structure(loss, predictions, targets)
  return sum_per_variable_losses(losses, per_variable_weights)


def _mean_preserving_batch(x: xarray.DataArray) -> xarray.DataArray:
  return x.mean([d for d in x.dims if d != 'batch'], skipna=False)


def sum_per_variable_losses(
    per_variable_losses: Mapping[str, xarray.DataArray],
    weights: Mapping[str, float],
) -> LossAndDiagnostics:
  """Weighted sum of per-variable losses."""
  if not set(weights.keys()).issubset(set(per_variable_losses.keys())):
    raise ValueError(
        'Passing a weight that does not correspond to any variable '
        f'{set(weights.keys())-set(per_variable_losses.keys())}')

  weighted_per_variable_losses = {
      name: loss * weights.get(name, 1)
      for name, loss in per_variable_losses.items()
  }
  total = xarray.concat(
      weighted_per_variable_losses.values(), dim='variable', join='exact').sum(
          'variable', skipna=False)
  return total, per_variable_losses  # pytype: disable=bad-return-type


def normalized_level_weights(data: xarray.DataArray) -> xarray.DataArray:
  """Weights proportional to pressure at each level."""
  level = data.coords['level']
  return level / level.mean(skipna=False)


def normalized_latitude_weights(data: xarray.DataArray) -> xarray.DataArray:
  """Weights based on latitude, roughly proportional to grid cell area.

  This method supports two use cases only (both for equispaced values):
  * Latitude values such that the closest value to the pole is at latitude
    (90 - d_lat/2), where d_lat is the difference between contiguous latitudes.
    For example: [-89, -87, -85, ..., 85, 87, 89]) (d_lat = 2)
    In this case each point with `lat` value represents a sphere slice between
    `lat - d_lat/2` and `lat + d_lat/2`, and the area of this slice would be
    proportional to:
    `sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)`, and
    we can simply omit the term `2 * sin(d_lat/2)` which is just a constant
    that cancels during normalization.
  * Latitude values that fall exactly at the poles.
    For example: [-90, -88, -86, ..., 86, 88, 90]) (d_lat = 2)
    In this case each point with `lat` value also represents
    a sphere slice between `lat - d_lat/2` and `lat + d_lat/2`,
    except for the points at the poles, that represent a slice between
    `90 - d_lat/2` and `90` or, `-90` and  `-90 + d_lat/2`.
    The areas of the first type of point are still proportional to:
    * sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)
    but for the points at the poles now is:
    * sin(90) - sin(90 - d_lat/2) = 2 * sin(d_lat/4) ^ 2
    and we will be using these weights, depending on whether we are looking at
    pole cells, or non-pole cells (omitting the common factor of 2 which will be
    absorbed by the normalization).

    It can be shown via a limit, or simple geometry, that in the small angles
    regime, the proportion of area per pole-point is equal to 1/8th
    the proportion of area covered by each of the nearest non-pole point, and we
    test for this in the test.

  Args:
    data: `DataArray` with latitude coordinates.
  Returns:
    Unit mean latitude weights.
  """
  latitude = data.coords['lat']

  if np.any(np.isclose(np.abs(latitude), 90.)):
    weights = _weight_for_latitude_vector_with_poles(latitude)
  else:
    weights = _weight_for_latitude_vector_without_poles(latitude)

  return weights / weights.mean(skipna=False)


def _weight_for_latitude_vector_without_poles(latitude):
  """Weights for uniform latitudes of the form [+-90-+d/2, ..., -+90+-d/2]."""
  delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
  if (not np.isclose(np.max(latitude), 90 - delta_latitude/2) or
      not np.isclose(np.min(latitude), -90 + delta_latitude/2)):
    raise ValueError(
        f'Latitude vector {latitude} does not start/end at '
        '+- (90 - delta_latitude/2) degrees.')
  return np.cos(np.deg2rad(latitude))


def _weight_for_latitude_vector_with_poles(latitude):
  """Weights for uniform latitudes of the form [+- 90, ..., -+90]."""
  delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
  if (not np.isclose(np.max(latitude), 90.) or
      not np.isclose(np.min(latitude), -90.)):
    raise ValueError(
        f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
  weights = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(delta_latitude/2))
  # The two checks above enough to guarantee that latitudes are sorted, so
  # the extremes are the poles
  weights[[0, -1]] = np.sin(np.deg2rad(delta_latitude/4)) ** 2
  return weights


def _check_uniform_spacing_and_get_delta(vector):
  diff = np.diff(vector)
  if not np.all(np.isclose(diff[0], diff)):
    raise ValueError(f'Vector {diff} is not uniformly spaced.')
  return diff[0]


# 이 파일은 GraphCast 모델에서 기상 예측 정확도를 측정하고 학습을 위한 손실(loss)을 계산하기 위한 losses.py 파일이야. 아래는 이 파일이 하는 역할을 핵심만 정리한 내용이야:

# ✅ 핵심 목적
# **예측값(predictions)**과 **정답(targets)**을 비교해서, 손실 함수(Loss Function) 값을 계산하고, 모델 학습에 사용할 수 있게 만드는 것.

# ✅ 주요 구성 요소
# 1. LossFunction (Protocol)
# 인터페이스 정의용 클래스야.


# 이 형식을 따르면 함수도 loss function으로 사용할 수 있어.


# input: predictions, targets


# output: (loss, diagnostics)
#  → loss: (batch,) shape의 평균 손실
#  → diagnostics: 추가 로그용 정보들 (예: 각 변수별 손실)



# 2. weighted_mse_per_level(...)
# 핵심 손실 함수임.


# **MSE(Mean Squared Error)**을 계산하되, 다음과 같은 가중치를 곱함:


# 위도(latitude) 기반 가중치 (normalized_latitude_weights)


# 압력 레벨(level) 기반 가중치 (normalized_level_weights)


# 변수별 가중치 (per_variable_weights)


# 이를 통해 지구 기후 특성과 물리적 의미를 반영한 loss 계산 가능



# 3. normalized_latitude_weights(...)
# 위도에 따른 셀 면적 차이 보정을 위해 가중치를 계산함
#  → 극지방은 셀이 작고, 적도는 셀이 크기 때문에 단순 평균을 내면 불균형이 생김
#  → 이를 보정해서 평균 loss가 물리적으로 의미 있도록 함


# 4. normalized_level_weights(...)
# 고도 레벨별(압력 기반) 가중치 적용
#  → 보통 낮은 레벨(지상)일수록 더 중요하거나 신뢰도 높음



# 5. sum_per_variable_losses(...)
# 여러 변수의 loss를 합칠 때 가중합을 계산
#  → 예: 온도는 1.0, 풍속은 0.5 가중치 줄 수 있음



# ✅ 예시 구조
# predictions = model_output  # xarray.Dataset
# targets = ground_truth_data  # xarray.Dataset

# loss, diagnostics = weighted_mse_per_level(
#     predictions, targets,
#     per_variable_weights={"z": 1.0, "u": 1.0, "v": 1.0, "t": 1.0}
# )


# ✅ 이 파일의 의미
# GraphCast는 지구 전역의 기상 데이터를 격자(grid)로 예측하는데, 단순 MSE는 위도/레벨에 따라 왜곡될 수 있어.
#  그래서 이 파일은:
# 물리적으로 공정하고 의미 있는 손실 계산


# 변수마다 다르게 가중치를 줄 수 있는 설정


# 학습 중 loss와 함께 모니터링할 진단값들 제공



# 필요하면 rollout loss랑 physics constraint loss로 이 구조 확장하는 것도 설명해줄 수 있어. 더 이어서 볼까?