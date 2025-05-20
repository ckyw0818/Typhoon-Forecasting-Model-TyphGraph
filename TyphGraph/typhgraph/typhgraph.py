from typing import Any, Callable, Mapping, Optional

import chex
import deep_typed_graph_net
import grid_mesh_connectivity
import icosahedral_mesh
from icosahedral_mesh import get_regionally_refined_mesh
import losses
import model_utils
import predictor_base
import typed_graph
import xarray_jax

import jax.numpy as jnp
import jraph
import numpy as np
import xarray

Kwargs = Mapping[str, Any]

GNN = Callable[[jraph.GraphsTuple], jraph.GraphsTuple]

# 🔧 1. 설정 관련 정의 (초반부)
# PRESSURE_LEVELS_*: ERA5, HRES, WeatherBench 등에서 사용되는 압력 레벨 정의.

# ALL_ATMOSPHERIC_VARS: 예측에 쓰이는 모든 대기 변수들.

# TARGET_*_VARS, FORCING_VARS, STATIC_VARS: 입력/출력 변수들을 의미 있고 그룹별로 나눠둔 것.

# TaskConfig: 입력/타겟/압력레벨/지속시간 등의 설정을 담는 데이터 클래스.

# ModelConfig: 모델 아키텍처에 대한 설정. 메시 크기, 레이어 수, 메시지 패싱 스텝 수 등. <COM_>


# https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
PRESSURE_LEVELS_ERA5_37 = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
    925, 950, 975, 1000)

# https://www.ecmwf.int/en/forecasts/datasets/set-i
PRESSURE_LEVELS_HRES_25 = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600,
    700, 800, 850, 900, 925, 950, 1000)

# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002203
PRESSURE_LEVELS_WEATHERBENCH_13 = (
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

PRESSURE_LEVELS = {
    13: PRESSURE_LEVELS_WEATHERBENCH_13,
    25: PRESSURE_LEVELS_HRES_25,
    37: PRESSURE_LEVELS_ERA5_37,
}

# """
# ㄴ> pressure Level임 <COM_>
# """

# The list of all possible atmospheric variables. Taken from:
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Table9
ALL_ATMOSPHERIC_VARS = (
    "potential_vorticity",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
    "vorticity",
    "divergence",
    "relative_humidity",
    "ozone_mass_mixing_ratio",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "fraction_of_cloud_cover",
)

TARGET_SURFACE_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "total_precipitation_6hr",
)
TARGET_SURFACE_NO_PRECIP_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
)
TARGET_ATMOSPHERIC_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
TARGET_ATMOSPHERIC_NO_W_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
)
EXTERNAL_FORCING_VARS = (
    "toa_incident_solar_radiation",
)
GENERATED_FORCING_VARS = (
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
)
FORCING_VARS = EXTERNAL_FORCING_VARS + GENERATED_FORCING_VARS
STATIC_VARS = (
    "geopotential_at_surface",
    "land_sea_mask",
)

# """
# ㄴ> variables를 의미 있는 그룹으로 나눔 <COM_>
# """

# 맞아, 정확해!

# 각 pressure level (예: 50, 100, ..., 1000 hPa) 마다 `TARGET_ATMOSPHERIC_VARS`에 포함된 **대기 변수들**이 모두 존재한다고 보면 돼. 즉:

# ---

# ### 🔁 예를 들어, `PRESSURE_LEVELS_ERA5_37`의 경우:

# ```python
# TARGET_ATMOSPHERIC_VARS = (
#     "temperature",
#     "geopotential",
#     "u_component_of_wind",
#     "v_component_of_wind",
#     "vertical_velocity",
#     "specific_humidity",
# )
# ```

# 이 6개의 변수가 다음 37개 pressure level에 대해 각각 존재함:

# ```
# 1, 2, 3, 5, 7, 10, 20, ..., 1000  (총 37개)
# ```

# ✅ 그러므로 총 대기 변수 개수는  
# **6 (변수 개수) × 37 (레벨 개수) = 222개의 채널 (혹은 feature dimension)**이 되는 구조야.

# ---

# ### 💡 반면, `TARGET_SURFACE_VARS`는 pressure level과 무관한 **지표면 변수**라서 그냥 개수만큼 한 번씩만 존재해.

# ---

# 즉, 최종 output의 변수 개수는:

# ```
# len(TARGET_SURFACE_VARS)
# + len(TARGET_ATMOSPHERIC_VARS) × len(PRESSURE_LEVELS)
# ```

# 으로 계산됨. 실제 `GraphCast.__init__()` 안에서 이렇게 계산되고 있어:

# ```python
# num_surface_vars = len(set(task_config.target_variables) - set(ALL_ATMOSPHERIC_VARS))
# num_atmospheric_vars = len(set(task_config.target_variables) & set(ALL_ATMOSPHERIC_VARS))
# num_outputs = (num_surface_vars + len(task_config.pressure_levels) * num_atmospheric_vars)
# ```

# 필요하면 output channel 수 체크할 수 있도록 간단한 유틸도 만들어줄게.


@chex.dataclass(frozen=True, eq=True)
class TaskConfig:
  """Defines inputs and targets on which a model is trained and/or evaluated."""
  input_variables: tuple[str, ...]
  # Target variables which the model is expected to predict.
  target_variables: tuple[str, ...]
  forcing_variables: tuple[str, ...]
  pressure_levels: tuple[int, ...]
  input_duration: str

TASK = TaskConfig(
    input_variables=(
        TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS + FORCING_VARS +
        STATIC_VARS),
    target_variables=TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS,
    forcing_variables=FORCING_VARS,
    pressure_levels=PRESSURE_LEVELS_ERA5_37,
    input_duration="12h",
)
TASK_13 = TaskConfig(
    input_variables=(
        TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS + FORCING_VARS +
        STATIC_VARS),
    target_variables=TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS,
    forcing_variables=FORCING_VARS,
    pressure_levels=PRESSURE_LEVELS_WEATHERBENCH_13,
    input_duration="12h",
)
TASK_13_PRECIP_OUT = TaskConfig(
    input_variables=(
        TARGET_SURFACE_NO_PRECIP_VARS + TARGET_ATMOSPHERIC_VARS + FORCING_VARS +
        STATIC_VARS),
    target_variables=TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS,
    forcing_variables=FORCING_VARS,
    pressure_levels=PRESSURE_LEVELS_WEATHERBENCH_13,
    input_duration="12h",
)

# """
# ㄴ> 입력/타겟/압력레벨/지속시간등을 다루는 데이터 클래스 <COM_>
# """


@chex.dataclass(frozen=True, eq=True)
class ModelConfig:
  """Defines the architecture of the GraphCast neural network architecture.

  Properties:
    resolution: The resolution of the data, in degrees (e.g. 0.25 or 1.0).
    mesh_size: How many refinements to do on the multi-mesh.
    gnn_msg_steps: How many Graph Network message passing steps to do.
    latent_size: How many latent features to include in the various MLPs.
    hidden_layers: How many hidden layers for each MLP.
    radius_query_fraction_edge_length: Scalar that will be multiplied by the
        length of the longest edge of the finest mesh to define the radius of
        connectivity to use in the Grid2Mesh graph. Reasonable values are
        between 0.6 and 1. 0.6 reduces the number of grid points feeding into
        multiple mesh nodes and therefore reduces edge count and memory use, but
        1 gives better predictions.
    mesh2grid_edge_normalization_factor: Allows explicitly controlling edge
        normalization for mesh2grid edges. If None, defaults to max edge length.
        This supports using pre-trained model weights with a different graph
        structure to what it was trained on.
  """
  resolution: float
  # 전 지구 균일 subdivide 깊이
  mesh_size_coarse: int
  # region_bbox 안을 최종 subdivide할 깊이
  mesh_size_fine: int
  # (lat_min, lat_max, lon_min, lon_max)
  region_bbox: tuple[float, float, float, float] #EDIT
  latent_size: int
  gnn_msg_steps: int
  hidden_layers: int
  radius_query_fraction_edge_length: float
  mesh2grid_edge_normalization_factor: Optional[float] = None

"""
ㄴ> 모델 구조 설명 <COM_>
"""

@chex.dataclass(frozen=True, eq=True)
class CheckPoint:
  params: dict[str, Any]
  model_config: ModelConfig
  task_config: TaskConfig
  description: str
  license: str


# """
# 🧠 2. GraphCast 클래스
# 🌐 구조 요약
# [Grid] --grid2mesh--> [Mesh] --mesh_gnn--> [Mesh] --mesh2grid--> [Grid (Output)]
# 📦 내부 GNN들
# _grid2mesh_gnn: grid에서 mesh로 정보를 전달하는 1-step GNN

# _mesh_gnn: mesh 내부에서 여러 번 message passing (기본 GNN 연산)

# _mesh2grid_gnn: mesh에서 다시 grid로 정보를 돌려주는 GNN <COM_>

# """

# 여기서, grid2mesh가 encoding이고 meshgnn은 mesh내에서 message passing이고 mesh2grid가 decodoing임

class GraphCast(predictor_base.Predictor):
  """GraphCast Predictor.

  The model works on graphs that take into account:
  * Mesh nodes: nodes for the vertices of the mesh.
  * Grid nodes: nodes for the points of the grid.
  * Nodes: When referring to just "nodes", this means the joint set of
    both mesh nodes, concatenated with grid nodes.

  The model works with 3 graphs:
  * Grid2Mesh graph: Graph that contains all nodes. This graph is strictly
    bipartite with edges going from grid nodes to mesh nodes using a
    fixed radius query. The grid2mesh_gnn will operate in this graph. The output
    of this stage will be a latent representation for the mesh nodes, and a
    latent representation for the grid nodes.
  * Mesh graph: Graph that contains mesh nodes only. The mesh_gnn will
    operate in this graph. It will update the latent state of the mesh nodes
    only.
  * Mesh2Grid graph: Graph that contains all nodes. This graph is strictly
    bipartite with edges going from mesh nodes to grid nodes such that each grid
    nodes is connected to 3 nodes of the mesh triangular face that contains
    the grid points. The mesh2grid_gnn will operate in this graph. It will
    process the updated latent state of the mesh nodes, and the latent state
    of the grid nodes, to produce the final output for the grid nodes.

  The model is built on top of `TypedGraph`s so the different types of nodes and
  edges can be stored and treated separately.

  """

  def __init__(self, model_config: ModelConfig, task_config: TaskConfig):
    """Initializes the predictor."""
    
    # """
    # 🛠️ 3. init 메서드
    # icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(...): icosahedron 기반 메시 생성

    # _grid2mesh_gnn, _mesh_gnn, _mesh2grid_gnn은 모두 DeepTypedGraphNet을 기반으로 한 GNN임.

    # 각각은 typed_graph.TypedGraph를 입력으로 받아 메시지 전달과 업데이트 수행.

    # 예측해야 하는 출력 변수 개수를 계산하여 mesh2grid GNN의 출력 크기 설정. <COM_>


    # """
    self._spatial_features_kwargs = dict(
        add_node_positions=False,
        add_node_latitude=True,
        add_node_longitude=True,
        add_relative_positions=True,
        relative_longitude_local_coordinates=True,
        relative_latitude_local_coordinates=True,
    )

    # Specification of the multimesh.
    self._meshes = get_regionally_refined_mesh(
            base_splits   = model_config.mesh_size_coarse,
            region_splits = model_config.mesh_size_fine,
            region_bbox   = model_config.region_bbox) #EDIT

    # Encoder, which moves data from the grid to the mesh with a single message
    # passing step.
    self._grid2mesh_gnn = deep_typed_graph_net.DeepTypedGraphNet(
        embed_nodes=True,  # Embed raw features of the grid and mesh nodes.
        embed_edges=True,  # Embed raw features of the grid2mesh edges.
        edge_latent_size=dict(grid2mesh=model_config.latent_size),
        node_latent_size=dict(
            mesh_nodes=model_config.latent_size,
            grid_nodes=model_config.latent_size),
        mlp_hidden_size=model_config.latent_size,
        mlp_num_hidden_layers=model_config.hidden_layers,
        num_message_passing_steps=1,
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=True,
        aggregate_normalization=None,
        name="grid2mesh_gnn",
    )

    # Processor, which performs message passing on the multi-mesh.
    self._mesh_gnn = deep_typed_graph_net.DeepTypedGraphNet(
        embed_nodes=False,  # Node features already embdded by previous layers.
        embed_edges=True,  # Embed raw features of the multi-mesh edges.
        node_latent_size=dict(mesh_nodes=model_config.latent_size),
        edge_latent_size=dict(mesh=model_config.latent_size),
        mlp_hidden_size=model_config.latent_size,
        mlp_num_hidden_layers=model_config.hidden_layers,
        num_message_passing_steps=model_config.gnn_msg_steps,
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=False,
        name="mesh_gnn",
    )

    num_surface_vars = len(
        set(task_config.target_variables) - set(ALL_ATMOSPHERIC_VARS))
    num_atmospheric_vars = len(
        set(task_config.target_variables) & set(ALL_ATMOSPHERIC_VARS))
    num_outputs = (num_surface_vars +
                   len(task_config.pressure_levels) * num_atmospheric_vars)

    # Decoder, which moves data from the mesh back into the grid with a single
    # message passing step.
    self._mesh2grid_gnn = deep_typed_graph_net.DeepTypedGraphNet(
        # Require a specific node dimensionaly for the grid node outputs.
        node_output_size=dict(grid_nodes=num_outputs),
        embed_nodes=False,  # Node features already embdded by previous layers.
        embed_edges=True,  # Embed raw features of the mesh2grid edges.
        edge_latent_size=dict(mesh2grid=model_config.latent_size),
        node_latent_size=dict(
            mesh_nodes=model_config.latent_size,
            grid_nodes=model_config.latent_size),
        mlp_hidden_size=model_config.latent_size,
        mlp_num_hidden_layers=model_config.hidden_layers,
        num_message_passing_steps=1,
        use_layer_norm=True,
        include_sent_messages_in_node_update=False,
        activation="swish",
        f32_aggregation=False,
        name="mesh2grid_gnn",
    )

    # Obtain the query radius in absolute units for the unit-sphere for the
    # grid2mesh model, by rescaling the `radius_query_fraction_edge_length`.
    self._query_radius = (_get_max_edge_distance(self._finest_mesh)
                          * model_config.radius_query_fraction_edge_length)
    self._mesh2grid_edge_normalization_factor = (
        model_config.mesh2grid_edge_normalization_factor
    )

    # Other initialization is delayed until the first call (`_maybe_init`)
    # when we get some sample data so we know the lat/lon values.
    self._initialized = False

    # A "_init_mesh_properties":
    # This one could be initialized at init but we delay it for consistency too.
    self._num_mesh_nodes = None  # num_mesh_nodes
    self._mesh_nodes_lat = None  # [num_mesh_nodes]
    self._mesh_nodes_lon = None  # [num_mesh_nodes]

    # A "_init_grid_properties":
    self._grid_lat = None  # [num_lat_points]
    self._grid_lon = None  # [num_lon_points]
    self._num_grid_nodes = None  # num_lat_points * num_lon_points
    self._grid_nodes_lat = None  # [num_grid_nodes]
    self._grid_nodes_lon = None  # [num_grid_nodes]

    # A "_init_{grid2mesh,processor,mesh2grid}_graph"
    self._grid2mesh_graph_structure = None
    self._mesh_graph_structure = None
    self._mesh2grid_graph_structure = None

  @property
  def _finest_mesh(self):
    return self._meshes[-1]

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               is_training: bool = False,
               ) -> xarray.Dataset:
    self._maybe_init(inputs)

# 좋아, `__call__` 메서드에서 GraphCast가 forward pass를 어떻게 수행하는지 **한 줄씩 따라가며 아주 자세히** 설명해볼게. 이 부분은 모델이 실제로 예측하는 **핵심 파이프라인**이니까 아주 중요해.

# ---

# ## 🚀 `__call__` 메서드 정밀 해부

# ```python
# def __call__(self,
#              inputs: xarray.Dataset,
#              targets_template: xarray.Dataset,
#              forcings: xarray.Dataset,
#              is_training: bool = False,
#              ) -> xarray.Dataset:
# ```

# ### 📥 Step 1: 입력 초기화 및 좌표 관련 메쉬 세팅
# ```python
# self._maybe_init(inputs)
# ```
# - 최초 1회에만 실행됨.
# - grid 위도/경도, mesh의 vertex 좌표, edge 연결 정보 등 **위치 기반 구조** 초기화.

# ---

# ### 🧾 Step 2: Grid 데이터를 벡터로 정리
# ```python
# grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)
# ```
# - `(batch, time, lat, lon, level, var)` 형식의 xarray → `[num_grid_nodes, batch, channels]` 로 변환.
# - 채널에는 `TARGET_*_VARS`, `FORCING_VARS`, `STATIC_VARS`의 값들이 포함됨.

# ---

# ### 🔁 Step 3: Grid → Mesh (Encoder)
# ```python
# (latent_mesh_nodes, latent_grid_nodes) = self._run_grid2mesh_gnn(grid_node_features)
# ```
# - **grid2mesh_gnn**에서 구조적 feature (위도, 경도 등)와 raw input을 합쳐서 message passing 1회.
# - output:
#   - `latent_mesh_nodes`: [num_mesh_nodes, batch, latent_dim]
#   - `latent_grid_nodes`: [num_grid_nodes, batch, latent_dim]
# - 이 단계는 **인코더 역할**을 함.

# ---

# ### 💡 Step 4: Mesh 내부 message passing (Processor)
# ```python
# updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)
# ```
# - **mesh_gnn**에서 mesh의 face 기반 edge들을 따라 여러 번 message passing.
# - output은 업데이트된 `latent_mesh_nodes`.
# - 이 부분이 GNN의 핵심 처리부. (multi-step message propagation)

# ---

# ### 🔃 Step 5: Mesh → Grid (Decoder)
# ```python
# output_grid_nodes = self._run_mesh2grid_gnn(
#     updated_latent_mesh_nodes, latent_grid_nodes)
# ```
# - mesh의 latent state와 초기 grid latent를 함께 사용해서 grid 예측을 생성.
# - `output_grid_nodes`: [num_grid_nodes, batch, output_dim]
#   - output_dim = surface vars 수 + pressure_levels × atm vars 수
# - 즉, 이게 **디코더 역할**.

# ---

# ### 🎁 Step 6: Grid 예측값 → xarray Dataset 변환
# ```python
# return self._grid_node_outputs_to_prediction(output_grid_nodes, targets_template)
# ```
# - `[grid_node, batch, output_dim]` → `(batch, time, lat, lon, level, var)` xarray.Dataset으로 복원.
# - `targets_template`를 따라 차원 및 변수 매핑.

# ---

# ## ✅ 전체 요약: 개념 흐름

# ```
# INPUTS (xarray)
#   ↓
# [Grid] → grid2mesh_gnn → [Mesh, Grid latent]
#   ↓
# [Mesh] → mesh_gnn → [Updated Mesh latent]
#   ↓
# [Grid] ← mesh2grid_gnn ← [Updated Mesh + Grid latent]
#   ↓
# OUTPUT (xarray)
# ```

# ---

# 필요하면 각 `_run_XXX_gnn` 함수 안에서 어떻게 데이터가 처리되는지도 구조적으로 뜯어볼 수 있어. 혹시 grid2mesh 내부 구조부터 들어가볼까?



    # Convert all input data into flat vectors for each of the grid nodes.
    # xarray (batch, time, lat, lon, level, multiple vars, forcings)
    # -> [num_grid_nodes, batch, num_channels]
    grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)

    # Transfer data for the grid to the mesh,
    # [num_mesh_nodes, batch, latent_size], [num_grid_nodes, batch, latent_size]
    (latent_mesh_nodes, latent_grid_nodes
     ) = self._run_grid2mesh_gnn(grid_node_features)

    # Run message passing in the multimesh.
    # [num_mesh_nodes, batch, latent_size]
    updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)

    # Transfer data frome the mesh to the grid.
    # [num_grid_nodes, batch, output_size]
    output_grid_nodes = self._run_mesh2grid_gnn(
        updated_latent_mesh_nodes, latent_grid_nodes)

    # Conver output flat vectors for the grid nodes to the format of the output.
    # [num_grid_nodes, batch, output_size] ->
    # xarray (batch, one time step, lat, lon, level, multiple vars)
    return self._grid_node_outputs_to_prediction(
        output_grid_nodes, targets_template)

  def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      ) -> tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
    
# 📉 5. Loss 계산
# python
# 복사
# 편집
# def loss_and_predictions(inputs, targets, forcings)
# 손실은 losses.weighted_mse_per_level() 함수로 계산됨.

# 2m 온도는 중요하게 다루고, 바람/기압/강수는 상대적으로 낮은 가중치 사용<COM_>

    # Forward pass.
    predictions = self(
        inputs, targets_template=targets, forcings=forcings, is_training=True)
    # Compute loss.
    loss = losses.weighted_mse_per_level(
        predictions, targets,
        per_variable_weights={
            # Any variables not specified here are weighted as 1.0.
            # A single-level variable, but an important headline variable
            # and also one which we have struggled to get good performance
            # on at short lead times, so leaving it weighted at 1.0, equal
            # to the multi-level variables:
            "2m_temperature": 1.0,
            # New single-level variables, which we don't weight too highly
            # to avoid hurting performance on other variables.
            "10m_u_component_of_wind": 0.1,
            "10m_v_component_of_wind": 0.1,
            "mean_sea_level_pressure": 0.1,
            "total_precipitation_6hr": 0.1,
        })
    return loss, predictions  # pytype: disable=bad-return-type  # jax-ndarray

  def loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      ) -> predictor_base.LossAndDiagnostics:
    loss, _ = self.loss_and_predictions(inputs, targets, forcings)
    return loss  # pytype: disable=bad-return-type  # jax-ndarray

  def _maybe_init(self, sample_inputs: xarray.Dataset):
    """Inits everything that has a dependency on the input coordinates."""
    if not self._initialized:
      self._init_mesh_properties()
      self._init_grid_properties(
          grid_lat=sample_inputs.lat, grid_lon=sample_inputs.lon)
      self._grid2mesh_graph_structure = self._init_grid2mesh_graph()
      self._mesh_graph_structure = self._init_mesh_graph()
      self._mesh2grid_graph_structure = self._init_mesh2grid_graph()

      self._initialized = True

  def _init_mesh_properties(self):
    """Inits static properties that have to do with mesh nodes."""
    self._num_mesh_nodes = self._finest_mesh.vertices.shape[0]
    mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
        self._finest_mesh.vertices[:, 0],
        self._finest_mesh.vertices[:, 1],
        self._finest_mesh.vertices[:, 2])
    (
        mesh_nodes_lat,
        mesh_nodes_lon,
    ) = model_utils.spherical_to_lat_lon(
        phi=mesh_phi, theta=mesh_theta)
    # Convert to f32 to ensure the lat/lon features aren't in f64.
    self._mesh_nodes_lat = mesh_nodes_lat.astype(np.float32)
    self._mesh_nodes_lon = mesh_nodes_lon.astype(np.float32)

  def _init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray):
    """Inits static properties that have to do with grid nodes."""
    self._grid_lat = grid_lat.astype(np.float32)
    self._grid_lon = grid_lon.astype(np.float32)
    # Initialized the counters.
    self._num_grid_nodes = grid_lat.shape[0] * grid_lon.shape[0]

    # Initialize lat and lon for the grid.
    grid_nodes_lon, grid_nodes_lat = np.meshgrid(grid_lon, grid_lat)
    self._grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32)
    self._grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32)


# 📦 6. 그래프 구조 초기화
# _init_grid2mesh_graph
# Grid 노드와 mesh 노드를 radius query로 연결

# edge, node에 위치 관련 feature들을 추가 (model_utils.get_bipartite_graph_spatial_features() 이용)

# _init_mesh_graph
# mesh 간 edge를 구성하고 structural features 설정

# _init_mesh2grid_graph
# grid point가 어떤 triangle에 포함되는지로 mesh→grid edge 구성<COM_>

  def _init_grid2mesh_graph(self) -> typed_graph.TypedGraph:
    """Build Grid2Mesh graph."""

    # Create some edges according to distance between mesh and grid nodes.
    assert self._grid_lat is not None and self._grid_lon is not None
    (grid_indices, mesh_indices) = grid_mesh_connectivity.radius_query_indices(
        grid_latitude=self._grid_lat,
        grid_longitude=self._grid_lon,
        mesh=self._finest_mesh,
        radius=self._query_radius)

    # Edges sending info from grid to mesh.
    senders = grid_indices
    receivers = mesh_indices

    # Precompute structural node and edge features according to config options.
    # Structural features are those that depend on the fixed values of the
    # latitude and longitudes of the nodes.
    (senders_node_features, receivers_node_features,
     edge_features) = model_utils.get_bipartite_graph_spatial_features(
         senders_node_lat=self._grid_nodes_lat,
         senders_node_lon=self._grid_nodes_lon,
         receivers_node_lat=self._mesh_nodes_lat,
         receivers_node_lon=self._mesh_nodes_lon,
         senders=senders,
         receivers=receivers,
         edge_normalization_factor=None,
         **self._spatial_features_kwargs,
     )

    n_grid_node = np.array([self._num_grid_nodes])
    n_mesh_node = np.array([self._num_mesh_nodes])
    n_edge = np.array([mesh_indices.shape[0]])
    grid_node_set = typed_graph.NodeSet(
        n_node=n_grid_node, features=senders_node_features)
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=receivers_node_features)
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set}
    edges = {
        typed_graph.EdgeSetKey("grid2mesh", ("grid_nodes", "mesh_nodes")):
            edge_set
    }
    grid2mesh_graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes=nodes,
        edges=edges)
    return grid2mesh_graph

  def _init_mesh_graph(self) -> typed_graph.TypedGraph:
    """Build Mesh graph."""
    merged_mesh = icosahedral_mesh.merge_meshes(self._meshes)

    # Work simply on the mesh edges.
    senders, receivers = icosahedral_mesh.faces_to_edges(merged_mesh.faces)

    # Precompute structural node and edge features according to config options.
    # Structural features are those that depend on the fixed values of the
    # latitude and longitudes of the nodes.
    assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
    node_features, edge_features = model_utils.get_graph_spatial_features(
        node_lat=self._mesh_nodes_lat,
        node_lon=self._mesh_nodes_lon,
        senders=senders,
        receivers=receivers,
        **self._spatial_features_kwargs,
    )

    n_mesh_node = np.array([self._num_mesh_nodes])
    n_edge = np.array([senders.shape[0]])
    assert n_mesh_node == len(node_features)
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=node_features)
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"mesh_nodes": mesh_node_set}
    edges = {
        typed_graph.EdgeSetKey("mesh", ("mesh_nodes", "mesh_nodes")): edge_set
    }
    mesh_graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes=nodes,
        edges=edges)

    return mesh_graph

  def _init_mesh2grid_graph(self) -> typed_graph.TypedGraph:
    """Build Mesh2Grid graph."""

    # Create some edges according to how the grid nodes are contained by
    # mesh triangles.
    (grid_indices,
     mesh_indices) = grid_mesh_connectivity.in_mesh_triangle_indices(
         grid_latitude=self._grid_lat,
         grid_longitude=self._grid_lon,
         mesh=self._finest_mesh)

    # Edges sending info from mesh to grid.
    senders = mesh_indices
    receivers = grid_indices

    # Precompute structural node and edge features according to config options.
    assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
    (senders_node_features, receivers_node_features,
     edge_features) = model_utils.get_bipartite_graph_spatial_features(
         senders_node_lat=self._mesh_nodes_lat,
         senders_node_lon=self._mesh_nodes_lon,
         receivers_node_lat=self._grid_nodes_lat,
         receivers_node_lon=self._grid_nodes_lon,
         senders=senders,
         receivers=receivers,
         edge_normalization_factor=self._mesh2grid_edge_normalization_factor,
         **self._spatial_features_kwargs,
     )

    n_grid_node = np.array([self._num_grid_nodes])
    n_mesh_node = np.array([self._num_mesh_nodes])
    n_edge = np.array([senders.shape[0]])
    grid_node_set = typed_graph.NodeSet(
        n_node=n_grid_node, features=receivers_node_features)
    mesh_node_set = typed_graph.NodeSet(
        n_node=n_mesh_node, features=senders_node_features)
    edge_set = typed_graph.EdgeSet(
        n_edge=n_edge,
        indices=typed_graph.EdgesIndices(senders=senders, receivers=receivers),
        features=edge_features)
    nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set}
    edges = {
        typed_graph.EdgeSetKey("mesh2grid", ("mesh_nodes", "grid_nodes")):
            edge_set
    }
    mesh2grid_graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=np.array([1]), features=()),
        nodes=nodes,
        edges=edges)
    return mesh2grid_graph


# 🧩 7. Message Passing 함수들
# _run_grid2mesh_gnn: raw feature + structural feature를 concat해서 GNN에 넣고 latent 추출

# _run_mesh_gnn: latent된 mesh feature를 GNN을 통해 업데이트

# _run_mesh2grid_gnn: latent mesh와 grid를 받아 최종 예측 grid output 생성<COM_>

# 좋아, 지금 보는 **message passing 함수 3개**는 GraphCast의 인코더-프로세서-디코더 파이프라인에서 실제로 데이터를 흘려보내는 핵심 로직이야. 하나씩 상세하게 뜯어보자:

# ---

# ## 🧩 7. Message Passing 함수들 상세 설명

# ---

# ### 1️⃣ `_run_grid2mesh_gnn()`

# > **[Grid Input] → grid2mesh_gnn → [Mesh & Grid latent]**

# #### 🧪 입력
# - `grid_node_features`:  
#   shape = `[num_grid_nodes, batch, channels]`  
#   (raw feature: input 변수들 + forcings 등)

# #### ⚙️ 내부 처리
# ```python
# # Grid 노드: raw + structural feature concat
# new_grid_nodes = grid_nodes._replace(features=jnp.concatenate([...]))
# # Mesh 노드: dummy zeros + structural feature concat
# new_mesh_nodes = mesh_nodes._replace(features=jnp.concatenate([...]))
# ```

# - Grid는 실제 feature (온도, 바람, 지형 등)
# - Mesh는 dummy 0값 (학습 가능 구조 위해 placeholder)

# #### 💬 Edge feature 처리
# ```python
# new_edges = edges._replace(features=_add_batch_second_axis(...))
# ```
# - 모든 edge feature에 batch 차원 붙이기 (broadcast)

# #### 💡 메시지 패싱
# ```python
# output = self._grid2mesh_gnn(input_graph)
# ```
# - GNN 실행 → 각 노드 (mesh + grid)에 latent state 저장

# #### ✅ 출력
# ```python
# latent_mesh_nodes = output.nodes["mesh_nodes"].features
# latent_grid_nodes = output.nodes["grid_nodes"].features
# ```
# - shape = `[num_nodes, batch, latent_dim]`  
# - 이 결과가 다음 단계 (mesh_gnn)로 들어감

# ---

# ### 2️⃣ `_run_mesh_gnn()`

# > **[Mesh latent] → mesh_gnn → [Updated Mesh latent]**

# #### 🧪 입력
# - `latent_mesh_nodes`:  
#   shape = `[num_mesh_nodes, batch, latent_dim]`  
#   (이전 단계에서 얻은 mesh latent state)

# #### ⚙️ 내부 처리
# ```python
# # Structural edge feature에 batch 차원 붙이기
# new_edges = edges._replace(features=_add_batch_second_axis(...))
# nodes = nodes._replace(features=latent_mesh_nodes)
# ```

# #### 💡 메시지 패싱
# ```python
# output = self._mesh_gnn(input_graph)
# ```
# - **multi-step message passing** 진행 (기본값: 8회)

# #### ✅ 출력
# ```python
# return output.nodes["mesh_nodes"].features
# ```
# - shape = `[num_mesh_nodes, batch, latent_dim]`
# - 업데이트된 mesh latent state가 디코더로 전달됨

# ---

# ### 3️⃣ `_run_mesh2grid_gnn()`

# > **[Updated Mesh latent + Grid latent] → mesh2grid_gnn → [Grid Output]**

# #### 🧪 입력
# - `updated_latent_mesh_nodes`:  
#   `[num_mesh_nodes, batch, latent_dim]`
# - `latent_grid_nodes`:  
#   `[num_grid_nodes, batch, latent_dim]`

# #### ⚙️ 내부 처리
# ```python
# new_mesh_nodes = mesh_nodes._replace(features=updated_latent_mesh_nodes)
# new_grid_nodes = grid_nodes._replace(features=latent_grid_nodes)
# new_edges = edges._replace(features=_add_batch_second_axis(...))
# ```

# - 양쪽 노드(latent)를 넣고, edge에는 구조적 정보만 사용

# #### 💡 메시지 패싱
# ```python
# output_graph = self._mesh2grid_gnn(input_graph)
# output_grid_nodes = output_graph.nodes["grid_nodes"].features
# ```

# - GNN이 mesh → grid로 메시지 전달해서 **최종 예측 값 생성**

# #### ✅ 출력
# - shape = `[num_grid_nodes, batch, output_dim]`  
#   - output_dim = surface vars 수 + (atm var 수 × pressure level 수)

# ---

# ## ✅ 요약

# | 함수 이름 | 역할 | 주요 입력 | 주요 출력 |
# |-----------|------|-----------|-----------|
# | `_run_grid2mesh_gnn()` | 인코더 | Raw input + 구조적 feature | Grid + Mesh latent |
# | `_run_mesh_gnn()` | 프로세서 | Mesh latent | Updated Mesh latent |
# | `_run_mesh2grid_gnn()` | 디코더 | Updated Mesh latent + Grid latent | Grid 예측 값 |

# ---

# 필요하면 각 GNN (`DeepTypedGraphNet`) 안에서 어떻게 message passing step이 돌아가는지도 구조적으로 파고들 수 있어. 더 들어가볼까?

  def _run_grid2mesh_gnn(self, grid_node_features: chex.Array,
                         ) -> tuple[chex.Array, chex.Array]:
    """Runs the grid2mesh_gnn, extracting latent mesh and grid nodes."""

    # Concatenate node structural features with input features.
    batch_size = grid_node_features.shape[1]

    grid2mesh_graph = self._grid2mesh_graph_structure
    assert grid2mesh_graph is not None
    grid_nodes = grid2mesh_graph.nodes["grid_nodes"]
    mesh_nodes = grid2mesh_graph.nodes["mesh_nodes"]
    new_grid_nodes = grid_nodes._replace(
        features=jnp.concatenate([
            grid_node_features,
            _add_batch_second_axis(
                grid_nodes.features.astype(grid_node_features.dtype),
                batch_size)
        ],
                                 axis=-1))

    # To make sure capacity of the embedded is identical for the grid nodes and
    # the mesh nodes, we also append some dummy zero input features for the
    # mesh nodes.
    dummy_mesh_node_features = jnp.zeros(
        (self._num_mesh_nodes,) + grid_node_features.shape[1:],
        dtype=grid_node_features.dtype)
    new_mesh_nodes = mesh_nodes._replace(
        features=jnp.concatenate([
            dummy_mesh_node_features,
            _add_batch_second_axis(
                mesh_nodes.features.astype(dummy_mesh_node_features.dtype),
                batch_size)
        ],
                                 axis=-1))

    # Broadcast edge structural features to the required batch size.
    grid2mesh_edges_key = grid2mesh_graph.edge_key_by_name("grid2mesh")
    edges = grid2mesh_graph.edges[grid2mesh_edges_key]

    new_edges = edges._replace(
        features=_add_batch_second_axis(
            edges.features.astype(dummy_mesh_node_features.dtype), batch_size))

    input_graph = self._grid2mesh_graph_structure._replace(
        edges={grid2mesh_edges_key: new_edges},
        nodes={
            "grid_nodes": new_grid_nodes,
            "mesh_nodes": new_mesh_nodes
        })

    # Run the GNN.
    grid2mesh_out = self._grid2mesh_gnn(input_graph)
    latent_mesh_nodes = grid2mesh_out.nodes["mesh_nodes"].features
    latent_grid_nodes = grid2mesh_out.nodes["grid_nodes"].features
    return latent_mesh_nodes, latent_grid_nodes

  def _run_mesh_gnn(self, latent_mesh_nodes: chex.Array) -> chex.Array:
    """Runs the mesh_gnn, extracting updated latent mesh nodes."""

    # Add the structural edge features of this graph. Note we don't need
    # to add the structural node features, because these are already part of
    # the latent state, via the original Grid2Mesh gnn, however, we need
    # the edge ones, because it is the first time we are seeing this particular
    # set of edges.
    batch_size = latent_mesh_nodes.shape[1]

    mesh_graph = self._mesh_graph_structure
    assert mesh_graph is not None
    mesh_edges_key = mesh_graph.edge_key_by_name("mesh")
    edges = mesh_graph.edges[mesh_edges_key]

    # We are assuming here that the mesh gnn uses a single set of edge keys
    # named "mesh" for the edges and that it uses a single set of nodes named
    # "mesh_nodes"
    msg = ("The setup currently requires to only have one kind of edge in the"
           " mesh GNN.")
    assert len(mesh_graph.edges) == 1, msg

    new_edges = edges._replace(
        features=_add_batch_second_axis(
            edges.features.astype(latent_mesh_nodes.dtype), batch_size))

    nodes = mesh_graph.nodes["mesh_nodes"]
    nodes = nodes._replace(features=latent_mesh_nodes)

    input_graph = mesh_graph._replace(
        edges={mesh_edges_key: new_edges}, nodes={"mesh_nodes": nodes})

    # Run the GNN.
    return self._mesh_gnn(input_graph).nodes["mesh_nodes"].features

  def _run_mesh2grid_gnn(self,
                         updated_latent_mesh_nodes: chex.Array,
                         latent_grid_nodes: chex.Array,
                         ) -> chex.Array:
    """Runs the mesh2grid_gnn, extracting the output grid nodes."""

    # Add the structural edge features of this graph. Note we don't need
    # to add the structural node features, because these are already part of
    # the latent state, via the original Grid2Mesh gnn, however, we need
    # the edge ones, because it is the first time we are seeing this particular
    # set of edges.
    batch_size = updated_latent_mesh_nodes.shape[1]

    mesh2grid_graph = self._mesh2grid_graph_structure
    assert mesh2grid_graph is not None
    mesh_nodes = mesh2grid_graph.nodes["mesh_nodes"]
    grid_nodes = mesh2grid_graph.nodes["grid_nodes"]
    new_mesh_nodes = mesh_nodes._replace(features=updated_latent_mesh_nodes)
    new_grid_nodes = grid_nodes._replace(features=latent_grid_nodes)
    mesh2grid_key = mesh2grid_graph.edge_key_by_name("mesh2grid")
    edges = mesh2grid_graph.edges[mesh2grid_key]

    new_edges = edges._replace(
        features=_add_batch_second_axis(
            edges.features.astype(latent_grid_nodes.dtype), batch_size))

    input_graph = mesh2grid_graph._replace(
        edges={mesh2grid_key: new_edges},
        nodes={
            "mesh_nodes": new_mesh_nodes,
            "grid_nodes": new_grid_nodes
        })

    # Run the GNN.
    output_graph = self._mesh2grid_gnn(input_graph)
    output_grid_nodes = output_graph.nodes["grid_nodes"].features

    return output_grid_nodes


# 🔁 8. 입출력 변환
# _inputs_to_grid_node_features: xarray Dataset → flat [node, batch, channel]

# _grid_node_outputs_to_prediction: flat output → xarray Dataset<COM_>

# 좋아, 이제 마지막 핵심 파트인 **입출력 변환** 부분을 자세히 볼게.  
# GraphCast는 **xarray.Dataset** 형식으로 데이터를 받아서 처리하고, 예측도 xarray로 반환해야 하니까, 이 변환 과정이 굉장히 중요해.

# ---

# ## 🔁 8. 입출력 변환 함수

# ---

# ### 1️⃣ `_inputs_to_grid_node_features`

# > **xarray Dataset → `[num_grid_nodes, batch, channels]`**

# #### 🔸 입력
# ```python
# inputs: xarray.Dataset
# forcings: xarray.Dataset
# ```
# - 보통 차원이 이렇게 생겼음:  
#   `(batch, time, lat, lon, level, variable)`

# #### 🔸 처리 과정

# ```python
# stacked_inputs = model_utils.dataset_to_stacked(inputs)
# stacked_forcings = model_utils.dataset_to_stacked(forcings)
# stacked_inputs = xarray.concat([stacked_inputs, stacked_forcings], dim="channels")
# ```

# - `dataset_to_stacked`:  
#   여러 변수를 하나의 `channels` 축으로 합쳐서 `(batch, lat, lon, channels)` 형태로 만듦

# ```python
# grid_xarray_lat_lon_leading = model_utils.lat_lon_to_leading_axes(stacked_inputs)
# ```
# - lat/lon을 앞쪽 축으로 변경  
#   → `(lat, lon, batch, channels)`

# ```python
# return unwrap(...).reshape((-1,) + ...)
# ```
# - 마지막으로 `[lat * lon, batch, channels]` 형태로 평탄화  
#   → **`[num_grid_nodes, batch, channels]`**

# #### ✅ 최종 출력
# - 이 데이터가 `grid2mesh_gnn`의 입력으로 들어감

# ---

# ### 2️⃣ `_grid_node_outputs_to_prediction`

# > **`[num_grid_nodes, batch, output_dim]` → `xarray.Dataset`**

# #### 🔸 입력
# ```python
# grid_node_outputs: chex.Array  # [num_grid_nodes, batch, output_dim]
# targets_template: xarray.Dataset
# ```

# #### 🔸 처리 과정

# ```python
# grid_outputs_lat_lon_leading = grid_node_outputs.reshape(
#     grid_shape + grid_node_outputs.shape[1:]
# )
# ```
# - grid node 수 → (lat, lon) 모양으로 되돌림  
#   → shape: `(lat, lon, batch, output_dim)`

# ```python
# dims = ("lat", "lon", "batch", "channels")
# grid_xarray_lat_lon_leading = xarray_jax.DataArray(..., dims=dims)
# ```
# - 이걸 xarray DataArray로 변환

# ```python
# grid_xarray = model_utils.restore_leading_axes(grid_xarray_lat_lon_leading)
# ```
# - 축 순서를 원래대로 되돌림: `(batch, lat, lon, channels)`

# ```python
# return model_utils.stacked_to_dataset(grid_xarray.variable, targets_template)
# ```
# - channel 정보를 `target_template`를 보고 다시 `variable` + `level`로 분리해서  
#   `(batch, time, lat, lon, level, variable)` 형식의 **`xarray.Dataset`**으로 복원

# ---

# ## ✅ 요약

# | 함수 | 역할 | 입력 형식 | 출력 형식 |
# |------|------|-----------|------------|
# | `_inputs_to_grid_node_features` | Grid 입력 flatten | `xarray.Dataset` | `[num_grid_nodes, batch, channels]` |
# | `_grid_node_outputs_to_prediction` | 예측 결과 복원 | `[num_grid_nodes, batch, output_dim]` | `xarray.Dataset` |

# ---

# 이 부분은 GraphCast가 실제로 ERA5 데이터를 학습/예측할 수 있도록 해주는 I/O 변환기라고 보면 돼.  
# 혹시 `dataset_to_stacked`이나 `stacked_to_dataset`이 내부적으로 어떻게 작동하는지도 더 보고 싶어?

  def _inputs_to_grid_node_features(
      self,
      inputs: xarray.Dataset,
      forcings: xarray.Dataset,
      ) -> chex.Array:
    """xarrays -> [num_grid_nodes, batch, num_channels]."""

    # xarray `Dataset` (batch, time, lat, lon, level, multiple vars)
    # to xarray `DataArray` (batch, lat, lon, channels)
    stacked_inputs = model_utils.dataset_to_stacked(inputs)
    stacked_forcings = model_utils.dataset_to_stacked(forcings)
    stacked_inputs = xarray.concat(
        [stacked_inputs, stacked_forcings], dim="channels")

    # xarray `DataArray` (batch, lat, lon, channels)
    # to single numpy array with shape [lat_lon_node, batch, channels]
    grid_xarray_lat_lon_leading = model_utils.lat_lon_to_leading_axes(
        stacked_inputs)
    return xarray_jax.unwrap(grid_xarray_lat_lon_leading.data).reshape(
        (-1,) + grid_xarray_lat_lon_leading.data.shape[2:])

  def _grid_node_outputs_to_prediction(
      self,
      grid_node_outputs: chex.Array,
      targets_template: xarray.Dataset,
      ) -> xarray.Dataset:
    """[num_grid_nodes, batch, num_outputs] -> xarray."""

    # numpy array with shape [lat_lon_node, batch, channels]
    # to xarray `DataArray` (batch, lat, lon, channels)
    assert self._grid_lat is not None and self._grid_lon is not None
    grid_shape = (self._grid_lat.shape[0], self._grid_lon.shape[0])
    grid_outputs_lat_lon_leading = grid_node_outputs.reshape(
        grid_shape + grid_node_outputs.shape[1:])
    dims = ("lat", "lon", "batch", "channels")
    grid_xarray_lat_lon_leading = xarray_jax.DataArray(
        data=grid_outputs_lat_lon_leading,
        dims=dims)
    grid_xarray = model_utils.restore_leading_axes(grid_xarray_lat_lon_leading)

    # xarray `DataArray` (batch, lat, lon, channels)
    # to xarray `Dataset` (batch, one time step, lat, lon, level, multiple vars)
    return model_utils.stacked_to_dataset(
        grid_xarray.variable, targets_template)


def _add_batch_second_axis(data, batch_size):
  # data [leading_dim, trailing_dim]
  assert data.ndim == 2
  ones = jnp.ones([batch_size, 1], dtype=data.dtype)
  return data[:, None] * ones  # [leading_dim, batch, trailing_dim]


def _get_max_edge_distance(mesh):
  senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces)
  edge_distances = np.linalg.norm(
      mesh.vertices[senders] - mesh.vertices[receivers], axis=-1)
  return edge_distances.max()
