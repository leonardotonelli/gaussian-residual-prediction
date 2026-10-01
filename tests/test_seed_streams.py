"""Canonical RNG addresses, effective Torch collisions and evaluation isolation."""
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from iwm_replication.seed_streams import (
    SeedContext, SeedRegistry, context_from_config, preserve_rng, seeded_rng,
)


def test_canonical_vector_and_full_numpy_entropy():
    ctx = SeedContext(dataset="mpi3d", purpose="main-training", replication=1)
    key = ctx.key("data-source-order", partition="train", epoch=0)
    expected = ('{"campaign":"jepa-five-seed-20260924-v1","component":"data-source-order",'
                '"dataset":"mpi3d","derivation_version":"structured-sha256-v1","epoch":0,'
                '"module":null,"partition":"train","purpose":"main-training","rank":null,'
                '"replication":1,"restart":null,"sample_index":null,"step":null,"view":null}')
    assert key.canonical == expected
    raw = hashlib.sha256(expected.encode("utf-8")).digest()
    assert key.torch_seed == int.from_bytes(raw[:8], "big") & (2**63-1)
    entropy = [int.from_bytes(raw[i:i+4], "big") for i in range(0, 32, 4)]
    assert list(key.numpy_entropy) == entropy
    oracle = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    assert np.array_equal(ctx.numpy_rng("data-source-order", partition="train", epoch=0).integers(2**63, size=32),
                          oracle.integers(2**63, size=32))
    assert SeedContext.from_dict(json.loads(json.dumps(ctx.as_dict()))) == ctx


def test_components_replicates_purposes_and_ranks_do_not_alias():
    contexts = [SeedContext(dataset=d, purpose=p, replication=r)
                for d in ("mpi3d", "moving_mnist")
                for p, ids in (("main-training", range(1, 6)), ("development-training", (0,)),
                               ("software-smoke", (0,)), ("evaluation", (None,))) for r in ids]
    keys = [c.key(component) for c in contexts for component in ("latent", "loader", "sigreg")]
    assert len({k.digest for k in keys}) == len(keys)
    registry = SeedRegistry()
    for key in keys:
        registry.register(key)
    a, b = [SeedContext(dataset="mpi3d", purpose="main-training", replication=r) for r in (1, 2)]
    assert a.seed("data-source-order", epoch=1) != b.seed("data-source-order", epoch=0)
    assert a.seed("latent", rank=1) != b.seed("latent", rank=0)
    assert a.seed("sigreg", view="source") != a.seed("sigreg", view="target")
    with pytest.raises(ValueError, match="independent of rank"):
        a.key("data-source-order", rank=1)


def test_effective_cpu_collision_guard():
    # Supported Torch CPU generator ignores high seed bits despite a 64-bit API.
    a = torch.randn(16, generator=torch.Generator().manual_seed(1))
    b = torch.randn(16, generator=torch.Generator().manual_seed(1+2**32))
    assert torch.equal(a, b)
    registry = SeedRegistry()
    key = SimpleNamespace(torch_seed=1, digest="key-one")
    assert registry.register(key) == registry.register(key) == 1
    with pytest.raises(ValueError, match="effective Torch seed"):
        registry.register(SimpleNamespace(torch_seed=1+2**32, digest="key-two"))


def states():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def same(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])


def test_rng_isolation_and_exception_restore():
    before = states()
    with pytest.raises(RuntimeError):
        with preserve_rng():
            random.random(); np.random.rand(3); torch.randn(3)
            raise RuntimeError("simulated evaluator failure")
    same(before, states())
    results = []
    for _ in range(2):
        with seeded_rng(903):
            results.append((random.random(), np.random.rand(3), torch.randn(3)))
    assert results[0][0] == results[1][0]
    assert np.array_equal(results[0][1], results[1][1])
    assert torch.equal(results[0][2], results[1][2])
    same(before, states())


@pytest.mark.parametrize("updates", [{"replication": 0}, {"replication": 6}, {"replication": True},
    {"purpose": "evaluation", "replication": 1}, {"purpose": "development-training", "replication": 1},
    {"derivation_version": "unknown"}, {"dataset": "new-dataset"}])
def test_invalid_contexts(updates):
    args = dict(dataset="mpi3d", purpose="main-training", replication=1)
    args.update(updates)
    with pytest.raises(ValueError):
        SeedContext(**args)


def test_no_silent_version_fallback():
    assert context_from_config({}) is None
    for cfg in ({"seed_streams": None}, {"seed_streams": {"dataset": "mpi3d"}}):
        with pytest.raises(ValueError):
            context_from_config(cfg)
    ctx = SeedContext(dataset="mpi3d", purpose="main-training", replication=1)
    for indices in ({"role": "R1"}, {"step": 1.0}, {"step": True}, {"partition": 2}):
        with pytest.raises(ValueError):
            ctx.key("latent", **indices)
