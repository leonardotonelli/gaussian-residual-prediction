"""Campaign streams: worker/resume replay, paired roles, fixed and isolated evaluation."""
import random

import numpy as np
import pytest
import torch

from src.moving_mnist import GeneratorConfig, SourceState, sample_future_velocities
from src.moving_mnist_evaluation import (
    extract_features, fit_digit_probe, fit_ridge, forecast_bank, make_banks, state_hash)
from src.moving_mnist_full_training import FullTrainer, FullTrainingConfig, ROLES, model_spec
from src.moving_mnist_stream import OnlineClips, STREAM_VERSION, SEEDED_STREAM_VERSION
from src.moving_mnist_training import clip_loader
from src.seed_streams import SeedContext


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def context(replication=1):
    return SeedContext(dataset='moving_mnist', purpose='main-training', replication=replication)


def evaluation_context():
    return SeedContext(dataset='moving_mnist', purpose='evaluation', replication=None)


def data(ctx=None, size=12):
    images = np.zeros((12, 28, 28), dtype=np.uint8)
    for i in range(12):
        images[i, 6:22, 8 + i % 3:18 + i % 3] = 255
    return OnlineClips(images, np.arange(12) % 10, [f'train:{i}' for i in range(12)],
        split='train', seed=3, generator_config=GeneratorConfig(), identity_hash='fixture',
        size=size, seed_context=ctx)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def trainer(role, ctx):
    cfg = FullTrainingConfig(total_steps=3, batch_size=4)
    mc, _ = model_spec(role, {}, cfg)
    return FullTrainer(cfg, mc, torch.device('cpu'), seed_context=ctx)


def batches(dataset, workers=0, start=0):
    return list(clip_loader(dataset, batch_size=4, workers=workers, start=start,
                           seed=77 if dataset.seed_context is None else dataset.seed_context.seed('loader')))


def test_data_pairing_worker_invariance_and_replication_separation():
    baseline = data(context())
    expected = batches(baseline)
    equal(expected, batches(baseline, workers=2))
    equal(expected[1:], batches(baseline, workers=1, start=4))
    for role in ROLES:
        paired = data(context())
        equal(expected, batches(paired))
        assert set(paired[0]) == {'source', 'target', 'surrogate', 'sample_index'}
    assert [data(context(2)).record(i) for i in range(12)] != [baseline.record(i) for i in range(12)]
    assert baseline.contract['version'] == SEEDED_STREAM_VERSION
    assert data().contract['version'] == STREAM_VERSION
    assert 'seed_context' not in data().contract


@pytest.mark.parametrize('role', ROLES)
def test_all_roles_exact_resume_and_mixed_versions_rejected(tmp_path, role):
    dataset = data(context())
    batch = batches(dataset)
    full = trainer(role, context())
    full.update(batch[0])
    path = tmp_path / 'step1.pt'
    contract = {'data': dataset.contract}
    full.save(path, contract)
    expected = [full.update(b) for b in batch[1:]]
    resumed = trainer(role, context())
    torch.randn(71)
    resumed.load(path, contract)
    equal([resumed.update(b) for b in batches(dataset, workers=1, start=4)], expected)
    equal(resumed.model.state_dict(), full.model.state_dict())
    equal(resumed.optimizer.state_dict(), full.optimizer.state_dict())
    equal(resumed.scheduler.state_dict(), full.scheduler.state_dict())
    equal(resumed.residual_rng.get_state(), full.residual_rng.get_state())
    for a, b in zip(resumed.sigreg_generators or (), full.sigreg_generators or ()):
        equal(a.get_state(), b.get_state())
    for ctx in (context(2), None):
        with pytest.raises(ValueError, match='contract'):
            trainer(role, ctx).load(path, contract)
    legacy = trainer(role, None)
    legacy_path = tmp_path / 'legacy.pt'
    legacy.save(legacy_path, contract)
    with pytest.raises(ValueError, match='contract'):
        resumed.load(legacy_path, contract)


def test_training_component_streams_private_and_paired():
    batch = batches(data(context()))[0]
    signatures = {}
    for role in ROLES:
        before = rng_state()
        current = trainer(role, context())
        equal(rng_state(), before)
        private_before = [g.get_state().clone() for g in current.sigreg_generators or ()]
        current.update(batch)
        equal(rng_state(), before)
        signatures[role] = (current.residual_rng.get_state(),
                            [g.get_state() for g in current.sigreg_generators or ()])
        for old, new in zip(private_before, signatures[role][1]):
            assert not torch.equal(old, new)
    equal(signatures['R1'][0], signatures['S1'][0])
    equal(signatures['S0'][1], signatures['S1'][1])
    assert not torch.equal(signatures['S0'][1][0], signatures['S0'][1][1])
    assert not torch.equal(signatures['R1'][0], signatures['R0'][0])


def test_evaluation_banks_fixed_across_training_replications(monkeypatch):
    import src.moving_mnist_evaluation as evaluation
    toy = data()
    def load(root, manifest, split):
        return toy.images, toy.labels, [f'{split}:{i}' for i in range(12)]
    monkeypatch.setattr(evaluation, 'load_digits', load)
    banks = [make_banks(None, {'sha256': 'fixture'}, data(context(r)).contract,
        {'fit': 4, 'selection': 4, 'report': 4, 'queries': 4}, 9000,
        seed_context=evaluation_context()) for r in (1, 5)]
    for name in banks[0]:
        equal(banks[0][name].contract, banks[1][name].contract)
        equal([banks[0][name].record(i) for i in range(4)],
              [banks[1][name].record(i) for i in range(4)])
    assert banks[0]['queries'].record(0) != banks[0]['report'].record(0)
    with pytest.raises(ValueError, match='evaluation context'):
        make_banks(None, {}, {}, {}, 0, seed_context=context())


@pytest.mark.parametrize('ctx', [None, evaluation_context()])
def test_probe_rng_isolation_and_replay(ctx):
    x, labels = np.tile(np.eye(10), (4, 1)), np.tile(np.arange(10), 4)
    config = dict(epochs=3, patience=3, min_delta=1e-5, batch_size=20,
                  learning_rate=.1, weight_decay=0.)
    kwargs = dict(config=config, seed=44, device=torch.device('cpu'), seed_context=ctx)
    before = rng_state()
    result = fit_digit_probe(x, labels, x, labels, x, labels, **kwargs)
    equal(rng_state(), before)
    torch.randn(33)
    equal(result, fit_digit_probe(x, labels, x, labels, x, labels, **kwargs))


def test_fixed_forecast_truth_oracle_and_rng_isolation():
    ctx = evaluation_context()
    dataset = data(ctx, size=4)
    model = trainer('R1', context()).model.eval().requires_grad_(False)
    bank = extract_features(model, dataset, 4, torch.device('cpu'))
    head, diagnostics = fit_ridge(bank['target_projector'], bank['velocity'],
                                  bank['target_projector'], bank['velocity'], [1.])
    kwargs = dict(samples=4, futures=8, seed=77, device=torch.device('cpu'), batch_size=4,
                  train_norm_p99=diagnostics['train_standardized_norm_p99'], seed_context=ctx)
    before, frozen = rng_state(), state_hash(model)
    result = forecast_bank(model, dataset, head, **kwargs)
    equal(rng_state(), before)
    assert state_hash(model) == frozen
    equal(result, forecast_bank(model, dataset, head, **kwargs))
    assert not np.array_equal(result[2]['truth'][:, :4],
                               result[2]['prediction_independent_privileged_oracle'])
    state = SourceState(**dataset.record(0)['state'])
    truth, _ = sample_future_velocities(state, 8,
        ctx.numpy_rng('evaluation-truth', partition='train', sample_index=0), dataset.config)
    equal(result[2]['truth'][0], truth)
    with pytest.raises(ValueError, match='evaluation context'):
        forecast_bank(model, dataset, head, **{**kwargs, 'seed_context': context()})


def test_structured_runner_config_resume_and_frozen_load(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from src.moving_mnist_evaluation import file_hash, load_frozen
    from src.moving_mnist_full_training import SEEDED_FULL_TRAINING_VERSION
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('structured_runner', root / 'scripts/train_moving_mnist_full.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'load_identity_manifest', lambda p: {'sha256': 'fixture'})
    toy = data()
    def load(root, manifest, split):
        assert split == 'train'
        return toy.images, toy.labels, toy.ids
    monkeypatch.setattr(runner, 'load_digits', load)
    config = dict(status='training-integration-test-only', models={'R0': {}}, generator={},
        training=dict(total_steps=2, batch_size=4), identity_manifest='fixture.json',
        seed_streams=context().as_dict())
    kwargs = dict(role='R0', data_dir=tmp_path, device=torch.device('cpu'), workers=0)
    first = runner.run(config, output_dir=tmp_path / 'part1', stop_after=1, **kwargs)
    checkpoint = Path(first['checkpoints'][-1]['path'])
    state = torch.load(checkpoint, weights_only=True)
    assert state['version'] == SEEDED_FULL_TRAINING_VERSION
    assert state['seed_context'] == context().as_dict()
    final = runner.run(config, output_dir=tmp_path / 'part2', resume=checkpoint, **kwargs)
    endpoint = Path(final['checkpoints'][-1]['path'])
    before = rng_state()
    model, contract = load_frozen(endpoint, expected_sha256=file_hash(endpoint), expected_step=2,
                                  role='R0', repo=root, device=torch.device('cpu'))
    equal(rng_state(), before)
    assert not model.training
    assert contract['seed_context'] == context().as_dict()
    with pytest.raises(ValueError, match='versioned seed context'):
        runner.run({**config, 'seed_streams': None}, output_dir=tmp_path / 'bad', **kwargs)


