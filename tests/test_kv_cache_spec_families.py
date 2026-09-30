# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Family handling in the KV cache spec path: attention vs. recurrent (mamba).

Covers the dense per-family layer indexing that an attention+mamba hybrid needs
(its attention layers are a sparse subset of the decoder depth, unlike the
sliding/full hybrids the path was originally written for) and the recurrent
state allocation contract.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_tt_plugin.model_runner import (
    TTModelRunner,
    _dense_layer_indices,
    _split_groups_by_family,
)


def _attention_spec(block_size: int = 64) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )


def _mamba_spec(
    block_size: int = 64,
    shapes: tuple[tuple[int, ...], ...] = ((4, 128, 128), (3, 2048)),
    dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.bfloat16),
) -> MambaSpec:
    return MambaSpec(block_size=block_size, shapes=shapes, dtypes=dtypes)


def _group(layer_names, spec):
    return SimpleNamespace(layer_names=list(layer_names), kv_cache_spec=spec)


# One (4, 128, 128) fp32 recurrent-state page.
_MAMBA_PAGE_BYTES = 4 * 128 * 128 * 4


def _kv_cache_config(sizes_by_layer, num_blocks=9999):
    """Config stub whose ``kv_cache_tensors`` give each layer its own size.

    ``num_blocks`` is deliberately a decoy: the mamba path must derive its
    block count per layer from that layer's own buffer, never from this
    attention-group count.
    """
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            SimpleNamespace(shared_by=[name], size=size)
            for name, size in sizes_by_layer.items()
        ],
    )


# --- dense per-family indexing -------------------------------------------


def test_dense_layer_indices_is_identity_for_contiguous_layers():
    """Attention-only models (incl. sliding/full hybrids) are unaffected."""
    names = [f"model.layers.{i}.self_attn" for i in range(4)]

    assert _dense_layer_indices(names) == {
        "model.layers.0.self_attn": 0,
        "model.layers.1.self_attn": 1,
        "model.layers.2.self_attn": 2,
        "model.layers.3.self_attn": 3,
    }


def test_dense_layer_indices_densifies_sparse_attention_layers():
    """The attention+mamba case: decoder index 11 is attention layer 2."""
    names = [
        "model.layers.3.self_attn",
        "model.layers.7.self_attn",
        "model.layers.11.self_attn",
    ]

    assert _dense_layer_indices(names) == {
        "model.layers.3.self_attn": 0,
        "model.layers.7.self_attn": 1,
        "model.layers.11.self_attn": 2,
    }


def test_dense_layer_indices_orders_by_global_index_not_iteration_order():
    names = [
        "model.layers.11.self_attn",
        "model.layers.3.self_attn",
        "model.layers.7.self_attn",
    ]

    assert _dense_layer_indices(names)["model.layers.3.self_attn"] == 0
    assert _dense_layer_indices(names)["model.layers.11.self_attn"] == 2


# --- family split ---------------------------------------------------------


def test_split_groups_by_family_partitions_mamba_from_attention():
    attn = _group(["model.layers.0.self_attn"], _attention_spec())
    mamba = _group(["model.layers.1.linear_attn"], _mamba_spec())

    attention_groups, mamba_groups = _split_groups_by_family([attn, mamba, attn])

    assert attention_groups == [attn, attn]
    assert mamba_groups == [mamba]


def test_split_groups_by_family_handles_attention_only_config():
    attn = _group(["model.layers.0.self_attn"], _attention_spec())

    attention_groups, mamba_groups = _split_groups_by_family([attn])

    assert attention_groups == [attn]
    assert mamba_groups == []


# --- validation -----------------------------------------------------------


def test_validate_kv_cache_groups_accepts_mamba_spec():
    groups = [
        _group(["model.layers.0.self_attn"], _attention_spec()),
        _group(["model.layers.1.linear_attn"], _mamba_spec()),
    ]

    TTModelRunner._validate_kv_cache_groups(groups)


def test_validate_kv_cache_groups_rejects_unknown_spec():
    groups = [_group(["model.layers.0.self_attn"], object())]

    with pytest.raises(TypeError, match="MambaSpec"):
        TTModelRunner._validate_kv_cache_groups(groups)


def test_validate_kv_cache_groups_rejects_empty():
    with pytest.raises(ValueError, match="no groups"):
        TTModelRunner._validate_kv_cache_groups([])


# --- mamba buffer shapes --------------------------------------------------


def test_mamba_cache_shapes_page_indexes_each_buffer():
    """One buffer per (shape, dtype) pair, each prefixed with num_blocks.

    Mirrors upstream's GPU allocation, which builds ``(num_blocks, *shape)``
    per entry of ``zip(spec.shapes, spec.dtypes)``.
    """
    spec = _mamba_spec(
        shapes=((4, 128, 128), (3, 2048)),
        dtypes=(torch.float32, torch.bfloat16),
    )

    buffers = TTModelRunner._mamba_cache_shapes(spec, num_blocks=17)

    assert buffers == [
        ((17, 4, 128, 128), torch.float32),
        ((17, 3, 2048), torch.bfloat16),
    ]


def test_mamba_cache_shapes_does_not_shard_by_tp():
    """Recurrent state is replicated per rank, never sharded."""
    spec = _mamba_spec(shapes=((4, 128, 128),), dtypes=(torch.float32,))

    ((shape, _dtype),) = TTModelRunner._mamba_cache_shapes(spec, num_blocks=1)

    assert shape == (1, 4, 128, 128)


def test_mamba_cache_shapes_rejects_shape_dtype_mismatch():
    spec = _mamba_spec(shapes=((4, 128, 128), (3, 2048)), dtypes=(torch.float32,))

    with pytest.raises(ValueError, match="correspond pairwise"):
        TTModelRunner._mamba_cache_shapes(spec, num_blocks=1)


# --- per-mamba-layer specs and allocation ---------------------------------


def _runner_stub(num_mamba_layers: int, model=None):
    """Minimal stub for the attributes the mamba allocation path touches.

    ``SimpleNamespace`` does not bind descriptors, so the instance methods
    under test are attached as explicit partials over the stub itself;
    ``_mamba_cache_shapes`` is a staticmethod and needs no receiver.
    """
    stub = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_layers_by_block_type=lambda _pc, block_type: (
                num_mamba_layers if block_type == "mamba" else 0
            )
        ),
        parallel_config=SimpleNamespace(),
        model=model,
        _mamba_cache_shapes=TTModelRunner._mamba_cache_shapes,
    )
    stub._build_per_mamba_layer_specs = partial(
        TTModelRunner._build_per_mamba_layer_specs, stub
    )
    return stub


def test_build_per_mamba_layer_specs_indexes_densely():
    """Mamba layers at decoder indices 0,1,2,4 become mamba indices 0..3."""
    spec = _mamba_spec(shapes=((4, 128, 128),), dtypes=(torch.float32,))
    groups = [
        _group(
            [
                "model.layers.0.linear_attn",
                "model.layers.1.linear_attn",
                "model.layers.2.linear_attn",
                "model.layers.4.linear_attn",
            ],
            spec,
        )
    ]
    stub = _runner_stub(num_mamba_layers=4)
    config = _kv_cache_config(
        {name: _MAMBA_PAGE_BYTES * 5 for name in groups[0].layer_names}
    )

    per_layer = TTModelRunner._build_per_mamba_layer_specs(stub, config, groups)

    assert len(per_layer) == 4
    assert all(buf == [((5, 4, 128, 128), torch.float32)] for buf in per_layer)


def test_build_per_mamba_layer_specs_derives_blocks_per_layer_not_from_config():
    """A mamba page is a whole state snapshot; the attention block count would
    over-allocate by orders of magnitude, so it must not be used."""
    spec = _mamba_spec(shapes=((4, 128, 128),), dtypes=(torch.float32,))
    groups = [
        _group(["model.layers.0.linear_attn", "model.layers.1.linear_attn"], spec)
    ]
    stub = _runner_stub(num_mamba_layers=2)
    config = _kv_cache_config(
        {
            "model.layers.0.linear_attn": _MAMBA_PAGE_BYTES * 2,
            "model.layers.1.linear_attn": _MAMBA_PAGE_BYTES * 7,
        },
        num_blocks=9999,
    )

    per_layer = TTModelRunner._build_per_mamba_layer_specs(stub, config, groups)

    assert per_layer[0] == [((2, 4, 128, 128), torch.float32)]
    assert per_layer[1] == [((7, 4, 128, 128), torch.float32)]


def test_build_per_mamba_layer_specs_rejects_layer_with_no_tensor():
    spec = _mamba_spec(shapes=((4, 128, 128),), dtypes=(torch.float32,))
    groups = [_group(["model.layers.0.linear_attn"], spec)]
    stub = _runner_stub(num_mamba_layers=1)

    with pytest.raises(ValueError, match="No KVCacheTensor covers mamba layer"):
        TTModelRunner._build_per_mamba_layer_specs(stub, _kv_cache_config({}), groups)


def test_build_per_mamba_layer_specs_rejects_buffer_smaller_than_one_page():
    spec = _mamba_spec(shapes=((4, 128, 128),), dtypes=(torch.float32,))
    groups = [_group(["model.layers.0.linear_attn"], spec)]
    stub = _runner_stub(num_mamba_layers=1)
    config = _kv_cache_config({"model.layers.0.linear_attn": _MAMBA_PAGE_BYTES - 1})

    with pytest.raises(ValueError, match="smaller than one"):
        TTModelRunner._build_per_mamba_layer_specs(stub, config, groups)


def test_build_per_mamba_layer_specs_rejects_uncovered_layer():
    spec = _mamba_spec(shapes=((4, 128, 128),), dtypes=(torch.float32,))
    groups = [_group(["model.layers.0.linear_attn"], spec)]
    stub = _runner_stub(num_mamba_layers=2)
    config = _kv_cache_config({"model.layers.0.linear_attn": _MAMBA_PAGE_BYTES})

    with pytest.raises(ValueError, match="No KVCacheGroupSpec covers mamba"):
        TTModelRunner._build_per_mamba_layer_specs(stub, config, groups)


def test_allocate_mamba_caches_is_noop_without_mamba_groups():
    """Every TT model today declares no MambaSpec; nothing should change."""
    stub = _runner_stub(num_mamba_layers=0, model=object())

    result = TTModelRunner._allocate_mamba_caches(
        stub, SimpleNamespace(num_blocks=1), []
    )

    assert result is None


def test_allocate_mamba_caches_requires_model_hook():
    spec = _mamba_spec(shapes=((4, 128, 128),), dtypes=(torch.float32,))
    groups = [_group(["model.layers.0.linear_attn"], spec)]
    stub = _runner_stub(num_mamba_layers=1, model=object())

    config = _kv_cache_config({"model.layers.0.linear_attn": _MAMBA_PAGE_BYTES})

    with pytest.raises(NotImplementedError, match="allocate_mamba_cache_per_layer"):
        TTModelRunner._allocate_mamba_caches(stub, config, groups)


def test_allocate_mamba_caches_delegates_to_model_hook():
    spec = _mamba_spec(shapes=((4, 128, 128),), dtypes=(torch.float32,))
    groups = [_group(["model.layers.0.linear_attn"], spec)]
    received = {}

    class _Model:
        def allocate_mamba_cache_per_layer(self, per_layer_specs):
            received["specs"] = per_layer_specs
            return "allocated"

    stub = _runner_stub(num_mamba_layers=1, model=_Model())
    config = _kv_cache_config({"model.layers.0.linear_attn": _MAMBA_PAGE_BYTES * 3})

    result = TTModelRunner._allocate_mamba_caches(stub, config, groups)

    assert result == "allocated"
    assert received["specs"] == [[((3, 4, 128, 128), torch.float32)]]
