#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_typhgraph_example.py

typhgraph 모델 초기화 · 예측 · 손실/그래디언트 계산 예시 스크립트
"""

import functools
import jax
import haiku as hk
import numpy as np
import xarray as xr

import typhgraph
import casting, normalization, autoregressive, data_utils, checkpoint
import xarray_jax, xarray_tree

# ------------------------------------------------------------------------------
# 1) TyphGraph 래퍼 및 JAX 변환 함수 정의
# ------------------------------------------------------------------------------

def construct_wrapped_typhgraph(model_config, task_config,
                                diffs_stddev_by_level, mean_by_level, stddev_by_level):
    predictor = typhgraph.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level)
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor

@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_typhgraph(
        model_config, task_config,
        diffs_stddev_by_level, mean_by_level, stddev_by_level)
    return predictor(inputs,
                     targets_template=targets_template,
                     forcings=forcings)

@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    predictor = construct_wrapped_typhgraph(
        model_config, task_config,
        diffs_stddev_by_level, mean_by_level, stddev_by_level)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    # 배치 평균된 scalar로 변환
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, jax.random.PRNGKey(0),
            model_config, task_config, i, t, f)
        return loss, (diagnostics, next_state)
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
        _aux, has_aux=True)(params, state, inputs, targets, forcings)
    return loss, diagnostics, next_state, grads

# ------------------------------------------------------------------------------
# 2) 유틸리티 래퍼
# ------------------------------------------------------------------------------

def with_configs(fn, model_config, task_config):
    return functools.partial(fn, model_config=model_config, task_config=task_config)

def with_params(fn, params, state):
    return functools.partial(fn, params=params, state=state)

def drop_state(fn):
    return lambda **kw: fn(**kw)[0]

# ------------------------------------------------------------------------------
# 3) main() 함수: 설정 · 데이터 로드 · 초기화 · 실행
# ------------------------------------------------------------------------------

def main():
    # --- 3.1 모델 설정 및 체크포인트 로드 (또는 랜덤 초기화) ---
    ckpt_path = None  # None 으로 하면 랜덤 모델
    if ckpt_path is not None:
        ckpt = checkpoint.load(open(ckpt_path, "rb"), typhgraph.CheckPoint)
        params = ckpt.params
        state = {}
        model_config = ckpt.model_config
        task_config  = ckpt.task_config
        print("Loaded checkpoint:", ckpt.description)
    else:
        params = None
        state = {}
        # 임의의 (랜덤) 모델 구성 예시
        model_config = typhgraph.ModelConfig(
            resolution=1.0,
            mesh_size_coarse=4,
            mesh_size_fine=4,
            region_bbox=(-90, 90, 0, 360),
            latent_size=32,
            gnn_msg_steps=4,
            hidden_layers=1,
            radius_query_fraction_edge_length=0.6)
        task_config = typhgraph.TaskConfig(
            input_variables=typhgraph.TASK.input_variables,
            target_variables=typhgraph.TASK.target_variables,
            forcing_variables=typhgraph.TASK.forcing_variables,
            pressure_levels=typhgraph.PRESSURE_LEVELS[13],
            input_duration=typhgraph.TASK.input_duration)

    # --- 3.2 정규화 통계 로드 ---
    global diffs_stddev_by_level, mean_by_level, stddev_by_level
    diffs_stddev_by_level = xr.load_dataset("path/to/diffs_stddev_by_level.nc")
    mean_by_level          = xr.load_dataset("path/to/mean_by_level.nc")
    stddev_by_level        = xr.load_dataset("path/to/stddev_by_level.nc")

    # --- 3.3 데이터 로드 및 분할 ---
    ds = xr.open_dataset("path/to/example_batch.nc").load()
    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
        ds, target_lead_times=slice("6h","6h"),  # 예시: 한 스텝만
        **task_config.__dict__)
    print("Loaded data shapes:",
          train_inputs.sizes, train_targets.sizes, train_forcings.sizes)

    # --- 3.4 JIT 초기화 ---
    init_jitted = jax.jit(with_configs(run_forward.init,
        model_config=model_config, task_config=task_config))
    if params is None:
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=train_inputs,
            targets_template=train_targets,
            forcings=train_forcings)

    # --- 3.5 JIT 함수 정의 ---
    loss_apply_jitted = drop_state(with_params(
        jax.jit(with_configs(loss_fn.apply, model_config, task_config)),
        params, state))
    grads_jitted = with_params(
        jax.jit(with_configs(grads_fn, model_config, task_config)),
        params, state)
    predict_jitted = drop_state(with_params(
        jax.jit(with_configs(run_forward.apply, model_config, task_config)),
        params, state))

    # --- 3.6 단일 스텝 예측 실행 ---
    preds = predict_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=train_inputs,
        targets_template=train_targets * np.nan,
        forcings=train_forcings)
    print("Predictions:", preds)

    # --- 3.7 Loss 계산 ---
    loss = loss_apply_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=train_inputs,
        targets=train_targets,
        forcings=train_forcings)
    print(f"Loss: {float(loss):.6f}")

    # --- 3.8 Gradient 계산 ---
    loss_val, diagnostics, next_state, grads = grads_jitted(
        inputs=train_inputs,
        targets=train_targets,
        forcings=train_forcings)
    mean_grad = np.mean([np.abs(v).mean() for v in jax.tree_util.tree_leaves(grads)])
    print(f"Loss: {loss_val:.6f}, Mean |grad|: {mean_grad:.6e}")

if __name__ == "__main__":
    main()
