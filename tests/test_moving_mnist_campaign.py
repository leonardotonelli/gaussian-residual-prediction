"""Accepted campaign recipe and actual synthetic model/trainer invariants."""
import copy
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from src.moving_mnist_campaign import (
    CAMPAIGN_STATUS, RECIPE_VERSION, model_recipe_report, resolve_campaign_config,
    validate_campaign_contract,
)
from src.moving_mnist_full_training import (
    ROLES, SEEDED_FULL_TRAINING_VERSION, FullTrainer, FullTrainingConfig, model_spec,
)
from src.moving_mnist_shared import S0Config
from src.seed_streams import SeedContext


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def recipe():
    return yaml.safe_load((ROOT / 'config/campaigns/five_seed_v1/moving_mnist.yaml').read_text())


def context(rep=1, purpose='main-training'):
    return SeedContext('moving_mnist', purpose, rep)


def synthetic_trainer(role, *, steps=3):
    options, descriptor = resolve_campaign_config(recipe(), context())
    # This small synthetic fixture is deliberately not a campaign run or a
    # reduced scientific recipe; all model options come from the accepted path.
    cfg = FullTrainingConfig(total_steps=steps, batch_size=4)
    mc, _ = model_spec(role, options[role], cfg)
    return FullTrainer(cfg, mc, torch.device('cpu'), seed_context=context()), descriptor


def batch(step):
    gen = torch.Generator().manual_seed(101 + step)
    return {**{name: torch.rand(4, 1, 3, 64, 64, generator=gen)
               for name in ('source', 'target', 'surrogate')},
            'sample_index': torch.arange(4*step, 4*(step+1))}


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('rep,purpose', [(rep, 'main-training') for rep in range(1, 6)] +
                         [(0, 'development-training'), (0, 'software-smoke')])
def test_exact_recipe_bindings_support_all_replications_and_distinct_development(rep, purpose):
    cfg = recipe()
    before = copy.deepcopy(cfg)
    ctx = context(rep, purpose)
    options, descriptor = resolve_campaign_config(cfg, ctx)
    assert cfg == before
    assert descriptor['version'] == RECIPE_VERSION
    assert descriptor['manifest_binding']['seed_context_sha256'] == ctx.sha256
    assert descriptor['training']['total_steps'] == 75000
    assert descriptor['training']['batch_size'] == 128
    assert descriptor['schedule']['eta_min'] == 0
    assert options['S0']['sigreg_seed'] == options['S1']['sigreg_seed'] == ctx.seed('sigreg', view='source')
    assert descriptor['sigreg_stream_seeds']['source'] != descriptor['sigreg_stream_seeds']['target']
    for role in ROLES:
        assert descriptor['resolved_model_configs'][role]['role'] == role
    assert descriptor['source_config'] == cfg


@pytest.mark.parametrize('path,value', [
    (('training', 'total_steps'), 74999), (('training', 'batch_size'), 64),
    (('training', 'total_steps'), 75000.0), (('training', 'seed'), 1),
    (('training', 'learning_rate'), .001), (('training', 'weight_decay'), 0),
    (('models', 'R1', 'beta'), .002), (('models', 'R0', 'ema_decay'), .999),
    (('models', 'S0', 'sigreg_seed'), 4000), (('models', 'S1', 'scale_floor'), 1e-5),
    (('models', 'S0', 'sigreg_weight'), .2), (('models', 'R0', 'spatial_strides'), [1,2,2]),
    (('generator', 'gaussian_parameter'), 'variance'), (('generator', 'setting'), 'B'),
    (('generator', 'noise_factor'), 1.), (('identity_manifest',), 'other.json'),
    (('recipe_version',), 'unknown'), (('new_training_option',), True),
    (('status',), 'exploratory-full-training'),
])
def test_recipe_tampering_rejected_before_any_data_access_or_output(path, value, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('campaign_mm_runner', ROOT/'scripts/train_moving_mnist_full.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    def forbidden(*a, **kw):
        pytest.fail('Invalid campaign accessed data')
    monkeypatch.setattr(runner, 'load_identity_manifest', forbidden)
    cfg = recipe()
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        runner.run(cfg, role='R0', seed_context=context(), data_dir=tmp_path,
                   output_dir=tmp_path/'must-not-exist', device=torch.device('cpu'), workers=0)
    assert not (tmp_path/'must-not-exist').exists()


@pytest.mark.parametrize('ctx', [None, SeedContext('mpi3d', 'main-training', 1),
                               SeedContext('moving_mnist', 'evaluation', None),
                               SeedContext('moving_mnist', 'main-training', 1, campaign='other')])
def test_campaign_has_no_implicit_or_cross_dataset_seed_fallback(ctx):
    with pytest.raises(ValueError, match='context'):
        resolve_campaign_config(recipe(), ctx)


@pytest.mark.parametrize('role', ROLES)
def test_actual_campaign_options_architecture_capacity_optimizer_and_schedule(role):
    trainer, _ = synthetic_trainer(role)
    model = trainer.model
    encoder = model.online_encoder if role.startswith('R') else model.encoder
    report = model_recipe_report(model)
    assert report['parameters_by_module']['encoder'] == 2359008
    assert report['parameter_counts']['trainable'] == (4081384 if role.endswith('1') else 3673824)
    assert report['stochastic_extra_predictor_weights'] == (2048 if role.endswith('1') else 0)
    assert report['parameters_by_module']['prior'] == (137220 if role.endswith('1') else 0)
    assert report['parameters_by_module']['posterior'] == (268292 if role.endswith('1') else 0)
    convs = [m for m in encoder.modules() if isinstance(m, nn.Conv3d)]
    assert [m.out_channels for m in convs] == [32, 64, 128, 128, 256]
    assert [m.stride for m in convs] == [(1,2,2), (1,2,2), (1,1,1), (1,1,1), (1,2,2)]
    assert isinstance(encoder.projector[-1], nn.BatchNorm1d)
    assert isinstance(model.predictor[-1], nn.Linear)
    assert not model.predictor[-1].bias
    group, = trainer.optimizer.param_groups
    assert (group['betas'], group['eps'], group['weight_decay']) == ((.9,.999), 1e-8, 1e-4)
    assert {id(p) for p in group['params']} == {id(p) for p in model.parameters() if p.requires_grad}
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm3d)) and mod.weight.requires_grad:
            assert id(mod.weight) in {id(p) for p in group['params']}
            assert id(mod.bias) in {id(p) for p in group['params']}
    for step in range(3):
        metrics = trainer.update(batch(step))
        assert metrics['learning_rate'] == pytest.approx(1e-4*.5*(1+math.cos(math.pi*step/3)))
        counts = trainer.normalization_counts()
        assert counts and all(row['observed'] == row['expected'] for row in counts.values())
        if role.startswith('R'):
            assert model.ema_updates.item() == step+1
            assert all(p.grad is None for p in model.target_encoder.parameters())
    assert trainer.optimizer.param_groups[0]['lr'] == 0
    assert trainer.scheduler.last_epoch == 3


@pytest.mark.parametrize('role', ROLES)
def test_campaign_options_exact_replay_and_bn_checkpoint_guard(tmp_path, role):
    trainer, descriptor = synthetic_trainer(role)
    trainer.update(batch(0))
    contract = {'synthetic_fixture': True, 'accepted_recipe_sha256': descriptor['sha256']}
    path = tmp_path/'step1.pt'
    trainer.save(path, contract)
    expected = trainer.update(batch(1))
    resumed, _ = synthetic_trainer(role)
    resumed.load(path, contract)
    equal(resumed.update(batch(1)), expected)
    equal(resumed.model.state_dict(), trainer.model.state_dict())
    equal(resumed.optimizer.state_dict(), trainer.optimizer.state_dict())
    equal(resumed.scheduler.state_dict(), trainer.scheduler.state_dict())
    equal(resumed.residual_rng.get_state(), trainer.residual_rng.get_state())
    for a, b in zip(resumed.sigreg_generators or (), trainer.sigreg_generators or ()):
        equal(a.get_state(), b.get_state())
    encoder = resumed.model.online_encoder if role.startswith('R') else resumed.model.encoder
    encoder(batch(2)['source'])  # Accidental extra train-mode feature extraction.
    with pytest.raises(ValueError, match='BN forwards'):
        resumed.save(tmp_path/'invalid.pt', contract)
    assert not (tmp_path/'invalid.pt').exists()


def test_shared_model_batch_must_agree_before_construction():
    with pytest.raises(ValueError, match='batch sizes'):
        FullTrainer(FullTrainingConfig(batch_size=4), S0Config(batch_size=8), torch.device('cpu'))


def test_serialized_contract_revalidation_includes_recipe_manifest_and_model_capacity():
    cfg = recipe()
    options, descriptor = resolve_campaign_config(cfg, context())
    training = FullTrainingConfig(**cfg['training'])
    mc, _ = model_spec('R1', options['R1'], training)
    trainer = FullTrainer(training, mc, torch.device('cpu'), seed_context=context())
    contract = {'version': SEEDED_FULL_TRAINING_VERSION, 'purpose': CAMPAIGN_STATUS,
                'campaign_recipe': descriptor, 'model_recipe': model_recipe_report(trainer.model),
                'seed_context': context().as_dict(), 'model': asdict(mc), 'training': asdict(training),
                'train_data': {'generator': descriptor['generator'], 'seed_context': context().as_dict()}}
    serialized = json.loads(json.dumps(contract))
    validate_campaign_contract(serialized, model=trainer.model)
    for field in ('sha256', 'manifest_binding', 'source_config'):
        changed = copy.deepcopy(serialized)
        changed['campaign_recipe'][field] = {} if field != 'sha256' else 'wrong'
        with pytest.raises((ValueError, TypeError)):
            validate_campaign_contract(changed, model=trainer.model)
    changed = copy.deepcopy(serialized)
    changed['model_recipe']['parameters_by_module']['prior'] += 1
    with pytest.raises(ValueError, match='capacity'):
        validate_campaign_contract(changed, model=trainer.model)


def test_configured_context_cannot_disagree_with_the_resolved_run():
    cfg = recipe()
    cfg['seed_streams'] = context(2).as_dict()
    with pytest.raises(ValueError, match='Conflicting'):
        resolve_campaign_config(cfg, context(1))


@pytest.mark.parametrize('role', ROLES)
def test_recipe_capacity_report_survives_frozen_inference(role):
    trainer, _ = synthetic_trainer(role)
    expected = model_recipe_report(trainer.model)
    trainer.model.eval().requires_grad_(False)
    assert model_recipe_report(trainer.model) == expected


def test_campaign_identity_filename_cannot_hide_another_split(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('campaign_split_runner', ROOT/'scripts/train_moving_mnist_full.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'load_identity_manifest', lambda path: {'seed': 1, 'sha256': 'other'})
    def forbidden(*args, **kwargs):
        pytest.fail('Changed identity split accessed training pixels')
    monkeypatch.setattr(runner, 'load_digits', forbidden)
    with pytest.raises(ValueError, match='seed-0 split'):
        runner.run(recipe(), role='R0', seed_context=context(), data_dir=tmp_path,
                   output_dir=tmp_path/'must-not-exist', device=torch.device('cpu'), workers=0)
    assert not (tmp_path/'must-not-exist').exists()
