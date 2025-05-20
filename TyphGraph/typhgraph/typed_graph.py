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
"""Data-structure for storing graphs with typed edges and nodes."""

from typing import NamedTuple, Any, Union, Tuple, Mapping, TypeVar

ArrayLike = Union[Any]  # np.ndarray, jnp.ndarray, tf.tensor
ArrayLikeTree = Union[Any, ArrayLike]  # Nest of ArrayLike

_T = TypeVar('_T')


# All tensors have a "flat_batch_axis", which is similar to the leading
# axes of graph_tuples:
# * In the case of nodes this is simply a shared node and flat batch axis, with
# size corresponding to the total number of nodes in the flattened batch.
# * In the case of edges this is simply a shared edge and flat batch axis, with
# size corresponding to the total number of edges in the flattened batch.
# * In the case of globals this is simply the number of graphs in the flattened
# batch.

# All shapes may also have any additional leading shape "batch_shape".
# Options for building batches are:
# * Use a provided "flatten" method that takes a leading `batch_shape` and
#   it into the flat_batch_axis (this will be useful when using `tf.Dataset`
#   which supports batching into RaggedTensors, with leading batch shape even
#   if graphs have different numbers of nodes and edges), so the RaggedBatches
#   can then be converted into something without ragged dimensions that jax can
#   use.
# * Directly build a "flat batch" using a provided function for batching a list
#   of graphs (how it is done in `jraph`).



# ### 1. `NodeSet`
# ```python
# class NodeSet(NamedTuple):
#   n_node: ArrayLike
#   features: ArrayLikeTree
# ```
# - 하나의 **노드 타입**에 해당
# - `n_node`: 그래프별 노드 수
# - `features`: 노드 feature들을 담은 tensor or tree (예: [total_nodes, feature_dim])
class NodeSet(NamedTuple):
  """Represents a set of nodes."""
  n_node: ArrayLike  # [num_flat_graphs]
  features: ArrayLikeTree  # Prev. `nodes`: [num_flat_nodes] + feature_shape



# ### 2. `EdgesIndices`
# ```python
# class EdgesIndices(NamedTuple):
#   senders: ArrayLike
#   receivers: ArrayLike
# ```
# - **엣지의 연결 정보** (source → target)
# - 각 인덱스는 flat한 노드 인덱스 기준

class EdgesIndices(NamedTuple):
  """Represents indices to nodes adjacent to the edges."""
  senders: ArrayLike  # [num_flat_edges]
  receivers: ArrayLike  # [num_flat_edges]



# ### 3. `EdgeSet`
# ```python
# class EdgeSet(NamedTuple):
#   n_edge: ArrayLike
#   indices: EdgesIndices
#   features: ArrayLikeTree
# ```
# - 하나의 **엣지 타입**에 해당
# - `n_edge`: 그래프별 엣지 수
# - `indices`: senders와 receivers
# - `features`: 엣지 feature들

class EdgeSet(NamedTuple):
  """Represents a set of edges."""
  n_edge: ArrayLike  # [num_flat_graphs]
  indices: EdgesIndices
  features: ArrayLikeTree  # Prev. `edges`: [num_flat_edges] + feature_shape


# ### 4. `Context`
# ```python
# class Context(NamedTuple):
#   n_graph: ArrayLike
#   features: ArrayLikeTree
# ```
# - 전체 그래프에 대한 **전역 feature**
# - 예: 시간, 위치, 기타 메타데이터 등
class Context(NamedTuple):
  # `n_graph` always contains ones but it is useful to query the leading shape
  # in case of graphs without any nodes or edges sets.
  n_graph: ArrayLike  # [num_flat_graphs]
  features: ArrayLikeTree  # Prev. `globals`: [num_flat_graphs] + feature_shape



# ### 5. `EdgeSetKey`
# ```python
# class EdgeSetKey(NamedTuple):
#   name: str
#   node_sets: Tuple[str, str]
# ```
# - 엣지의 타입 이름과 연결된 노드 타입 명을 명시
# - 예: `('edge_temp_to_pressure', ('temp', 'pressure'))`
class EdgeSetKey(NamedTuple):
  name: str   # Name of the EdgeSet.

  # Sender node set name and receiver node set name connected by the edge set.
  node_sets: Tuple[str, str]



# ### 6. `TypedGraph`
# ```python
# class TypedGraph(NamedTuple):
#   context: Context
#   nodes: Mapping[str, NodeSet]
#   edges: Mapping[EdgeSetKey, EdgeSet]
# ```
# - 전체 그래프 구조를 담는 핵심 클래스
# - `nodes`: {node_type_name: NodeSet}
# - `edges`: {EdgeSetKey: EdgeSet}
# - 함수:
#   - `edge_key_by_name(name)`: name으로 EdgeSetKey 반환
#   - `edge_by_name(name)`: name으로 EdgeSet 반환

# ---
class TypedGraph(NamedTuple):
  """A graph with typed nodes and edges.

  A typed graph is made of a context, multiple sets of nodes and multiple
  sets of edges connecting those nodes (as indicated by the EdgeSetKey).
  """

  context: Context
  nodes: Mapping[str, NodeSet]
  edges: Mapping[EdgeSetKey, EdgeSet]

  def edge_key_by_name(self, name: str) -> EdgeSetKey:
    found_key = [k for k in self.edges.keys() if k.name == name]
    if len(found_key) != 1:
      raise KeyError("invalid edge key '{}'. Available edges: [{}]".format(
          name, ', '.join(x.name for x in self.edges.keys())))
    return found_key[0]

  def edge_by_name(self, name: str) -> EdgeSet:
    return self.edges[self.edge_key_by_name(name)]

# 이 코드는 `typedgraph.py`이며, GraphCast에서 사용하는 **Typed Graph 구조**를 정의하는 핵심 데이터 구조 파일이야. 아래에 주요 구성요소들을 설명해줄게:

# ---

# ## ✅ 전체 목적
# **`TypedGraph`**는 서로 다른 타입의 노드들과 엣지들을 다룰 수 있도록 만든 **구조화된 그래프 데이터 구조**야. 각 노드/엣지 타입을 분리해서 관리할 수 있고, Graph Neural Network에 유용하게 쓰이지.

# ---

# ## 🧱 주요 클래스 설명

# ### 1. `NodeSet`
# ```python
# class NodeSet(NamedTuple):
#   n_node: ArrayLike
#   features: ArrayLikeTree
# ```
# - 하나의 **노드 타입**에 해당
# - `n_node`: 그래프별 노드 수
# - `features`: 노드 feature들을 담은 tensor or tree (예: [total_nodes, feature_dim])

# ---

# ### 2. `EdgesIndices`
# ```python
# class EdgesIndices(NamedTuple):
#   senders: ArrayLike
#   receivers: ArrayLike
# ```
# - **엣지의 연결 정보** (source → target)
# - 각 인덱스는 flat한 노드 인덱스 기준

# ---

# ### 3. `EdgeSet`
# ```python
# class EdgeSet(NamedTuple):
#   n_edge: ArrayLike
#   indices: EdgesIndices
#   features: ArrayLikeTree
# ```
# - 하나의 **엣지 타입**에 해당
# - `n_edge`: 그래프별 엣지 수
# - `indices`: senders와 receivers
# - `features`: 엣지 feature들

# ---

# ### 4. `Context`
# ```python
# class Context(NamedTuple):
#   n_graph: ArrayLike
#   features: ArrayLikeTree
# ```
# - 전체 그래프에 대한 **전역 feature**
# - 예: 시간, 위치, 기타 메타데이터 등

# ---

# ### 5. `EdgeSetKey`
# ```python
# class EdgeSetKey(NamedTuple):
#   name: str
#   node_sets: Tuple[str, str]
# ```
# - 엣지의 타입 이름과 연결된 노드 타입 명을 명시
# - 예: `('edge_temp_to_pressure', ('temp', 'pressure'))`

# ---

# ### 6. `TypedGraph`
# ```python
# class TypedGraph(NamedTuple):
#   context: Context
#   nodes: Mapping[str, NodeSet]
#   edges: Mapping[EdgeSetKey, EdgeSet]
# ```
# - 전체 그래프 구조를 담는 핵심 클래스
# - `nodes`: {node_type_name: NodeSet}
# - `edges`: {EdgeSetKey: EdgeSet}
# - 함수:
#   - `edge_key_by_name(name)`: name으로 EdgeSetKey 반환
#   - `edge_by_name(name)`: name으로 EdgeSet 반환

# ---

# ## 📌 요약
# 이 구조는 GraphCast나 GNN에서 **노드 타입, 엣지 타입이 명확히 분리된 그래프**를 표현하기 위해 설계됐고, 특히 기상처럼 이기종 데이터 구조를 다루는 데 매우 적합해.

# 필요하면 예제 입력이나 내부 흐름도 설명해줄게.