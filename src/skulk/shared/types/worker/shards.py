from enum import Enum
from typing import TypeAlias, final

from pydantic import Field

from skulk.shared.models.llama_server_settings import LlamaServerSettings
from skulk.shared.models.model_cards import ModelCard
from skulk.utils.pydantic_ext import TaggedModel


class Sharding(str, Enum):
    Tensor = "Tensor"
    Pipeline = "Pipeline"


class BaseShardMetadata(TaggedModel):
    """
    Defines a specific shard of the model that is ready to be run on a device.
    Replaces previous `Shard` object.
    """

    model_card: ModelCard
    device_rank: int
    world_size: int

    # The backend tag (e.g. "llama_cpp-vulkan") the master resolved for THIS
    # node at placement time, intersecting the card's compatible_backends with
    # the node's advertised backends. Persisting it makes worker engine dispatch
    # deterministic from replicated state instead of a node-local re-probe, and
    # lets a card legitimately resolve to different engines per node on a
    # heterogeneous cycle. ``None`` means the master did not record one (e.g. it
    # lacked the node's resources at placement); the worker then falls back to
    # its local backend probe. See #330.
    resolved_backend: str | None = None
    llama_server_settings: LlamaServerSettings | None = Field(
        default=None,
        description="Node serving settings captured by placement for memory admission.",
    )

    # Error handling; equivalent to monkey-patch, but we can't monkey-patch runner.py
    # This is kinda annoying because it allocates memory in the ShardMetadata object. Can be rethought after Shanghai.
    immediate_exception: bool = False
    should_timeout: float | None = None

    start_layer: int = Field(ge=0)
    end_layer: int = Field(ge=0)
    n_layers: int = Field(ge=0)

    @property
    def is_first_layer(self) -> bool:
        return self.start_layer == 0

    @property
    def is_last_layer(self) -> bool:
        return self.end_layer == self.n_layers

    def __hash__(self) -> int:
        return hash(
            (
                self.model_card.model_id,
                self.start_layer,
                self.end_layer,
                self.n_layers,
                self.device_rank,
                self.world_size,
            )
        )


@final
class PipelineShardMetadata(BaseShardMetadata):
    """
    Pipeline parallelism shard meta.

    Layers are represented as a half-open interval [start_layer, end_layer),
    where start_layer is inclusive and end_layer is exclusive.
    """


@final
class CfgShardMetadata(BaseShardMetadata):
    """Shard metadata for CFG-parallel image generation models."""

    cfg_rank: int  # 0 = positive branch, 1 = negative branch
    cfg_world_size: int = 2

    # Pipeline-relative coordinates (computed at placement time)
    pipeline_rank: int  # rank within the pipeline group (0, 1, 2, ...)
    pipeline_world_size: int  # number of nodes per pipeline group


@final
class TensorShardMetadata(BaseShardMetadata):
    pass


@final
class RpcDonorShardMetadata(BaseShardMetadata):
    """Memory-donor shard of a multi-node llama.cpp RPC placement (#328).

    A donor node runs ``ggml-rpc-server`` and lends its GPU memory to the
    driver's ``llama-server --rpc``; llama.cpp distributes weights/KV across
    the pooled devices itself, so a donor holds NO Skulk-assigned layer range
    (``start_layer == end_layer == 0``) and never downloads or reads the model
    file. The degenerate layer range keeps the one-runner-per-node
    ``ShardAssignments`` invariant and ``BoundInstance.bound_shard`` working
    unchanged, while the distinct type is what the worker dispatches on: a
    donor runner's whole job is to serve ``ggml-rpc-server`` on the endpoint
    the placement stamped for it and report ready.
    """


ShardMetadata: TypeAlias = (
    PipelineShardMetadata
    | CfgShardMetadata
    | TensorShardMetadata
    | RpcDonorShardMetadata
)
