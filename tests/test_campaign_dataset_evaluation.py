"""Synthetic full evaluator paths, including frozen final reuse, without dataset reads."""
import importlib.util
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml

from iwm_replication.campaign_evaluation import EVALUATION_VERSION
from iwm_replication.moving_mnist import GeneratorConfig
from iwm_replication.moving_mnist_data import content_hash
from iwm_replication.moving_mnist_evaluation import file_hash, state_hash
from iwm_replication.moving_mnist_full_training import FullTrainer, FullTrainingConfig, model_spec
from iwm_replication.moving_mnist_stream import OnlineClips
from iwm_replication.seed_streams import SeedContext

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def config():
    cfg = yaml.safe_load((ROOT / 'config/campaigns/five_seed_v1/evaluation.yaml').read_text())
    mm = cfg['moving_mnist']
    mm.update(batch_size=4, sizes=dict(fit=4, selection=4, report=4, queries=2), samples=3, futures=4, latent_futures=3,
              ridge_alphas=[.1])
    mm['digit_probe'].update(epochs=1, patience=1, batch_size=4)
    mp = cfg['mpi3d']
    mp.update(batch_size=4, probe_counts=dict(fit=8, selection=6, report=6), readout_counts=[8,4,4],
              queries=8, quantiles=3, ridge_alphas=[.1])
    mp['probe'].update(maximum_epochs=1, lambda_grid=[0.], batch_size=4)
    return cfg


def fake_digits(split):
    pixels = np.zeros((16, 28, 28), dtype=np.uint8)
    for i in range(len(pixels)):
        pixels[i, 6:22, 8 + i % 3:18 + i % 3] = 255
    return pixels, np.arange(len(pixels)) % 10, [f'{split}:{i}' for i in range(len(pixels))]


@pytest.mark.parametrize('role', ['R0', 'R1', 'S0', 'S1'])
def test_moving_mnist_complete_development_then_frozen_final_replay(tmp_path, monkeypatch, role):
    import iwm_replication.campaign_moving_mnist_evaluation as campaign
    import iwm_replication.moving_mnist_evaluation as legacy
    context = SeedContext(dataset='moving_mnist', purpose='main-training', replication=1)
    cfg = FullTrainingConfig(total_steps=1, batch_size=4)
    mc, _ = model_spec(role, {}, cfg)
    model = FullTrainer(cfg, mc, torch.device('cpu'), seed_context=context).model.eval().requires_grad_(False)
    before = state_hash(model)
    manifest = {'sha256': 'synthetic-identities'}
    data = OnlineClips(*fake_digits('train'), split='train', seed=3, generator_config=GeneratorConfig(),
                      identity_hash=manifest['sha256'], size=4, seed_context=context)
    training = {'seed_context': context.as_dict(), 'train_data': data.contract,
                'purpose': 'five-seed-campaign-training', 'campaign_recipe': {'synthetic-loader-fixture': True}}
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'model_config': {'role': role}, 'training_config': {'total_steps': 1}}, checkpoint)
    digest = file_hash(checkpoint)
    monkeypatch.setattr(campaign, 'load_frozen', lambda *a, **kw: (model, training))
    monkeypatch.setattr(campaign, 'load_identity_manifest', lambda *a: manifest)
    calls = []
    def load(directory, identity_manifest, split):
        calls.append(split)
        return fake_digits(split)
    monkeypatch.setattr(legacy, 'load_digits', load)
    monkeypatch.setattr(campaign, 'load_digits', load)
    kwargs = dict(checkpoint=checkpoint, expected_sha256=digest, data_dir=tmp_path, device=torch.device('cpu'))
    dev = tmp_path / 'development'
    report = campaign.run(config(), output_dir=dev, **kwargs)
    assert report['frozen_state_unchanged'] and state_hash(model) == before
    assert set(calls) == {'train', 'development'}
    fitted = dev / 'fitted.pt'
    arrays = np.load(dev / 'physical_forecasts.npz')
    latent = np.load(dev / 'latent_forecasts.npz')
    assert arrays['truth'].shape == (2, 4, 2)
    assert latent['prior'].shape == (2, 3 if role.endswith('1') else 1, 128)
    assert not np.array_equal(arrays['truth'][:, :3], arrays['prediction_independent_privileged_oracle'])
    per_query = json.loads((dev / 'per_query.json').read_text())
    for name, scores in per_query['physical']['scores'].items():
        assert np.mean(scores['energy_score']) == pytest.approx(report['forecasts'][name]['strata']['all']['energy_score'])
    calls.clear()
    final = tmp_path / 'final'
    analysis = yaml.safe_load((ROOT / 'config/campaigns/five_seed_v1/analysis.yaml').read_text())
    final_manifest = campaign.query_regeneration_manifest('moving_mnist',
        SeedContext(dataset='moving_mnist', purpose='evaluation', replication=None), config()['moving_mnist'], 'final',
        {'identity_manifest_sha256': data.contract['identity_manifest_sha256'], 'generator': data.contract['generator']})
    analysis.update(frozen=True, protocol_sha256='a'*64, evaluation_contract_sha256=content_hash(config()),
                    final_bank_sha256={'moving_mnist': content_hash(final_manifest), 'mpi3d': 'b'*64})
    campaign.run(config(), output_dir=final, mode='final', fitted_artifacts=fitted,
        fitted_sha256=file_hash(fitted), frozen_protocol_sha256='a' * 64, analysis_contract=analysis, **kwargs)
    assert calls == ['test'] and state_hash(model) == before
    summary = json.loads((final / 'summary.json').read_text())
    assert summary['partition'] == 'final' and summary['purpose'] == 'main-training'
    assert summary['evaluation_contract_sha256'] == content_hash(config())
    assert np.isfinite(summary['metrics']['physical_forecast_energy_score'])
    assert not (final / 'fitted.pt').exists()
    assert not (final / 'features_fit_source.npz').exists()


def matched_config(role):
    from iwm_replication.mpi3d_byol import resolve_matched_config
    cfg = yaml.safe_load((ROOT / 'config/campaigns/five_seed_v1/mpi3d.yaml').read_text())
    cfg['model'].update(vit_dim=16, vit_depth=1, vit_heads=2, projector_dim=8, head_hidden_dim=16)
    cfg['data'].update(batch_size=4, num_workers=0)
    cfg['train'].update(epochs=1, warmup_epochs=0, transition_samples_per_epoch=4, checkpoint_every=1)
    return resolve_matched_config(cfg, role=role, replication=0, purpose='software-smoke')


@pytest.mark.parametrize('role', ['R0', 'R1', 'S0', 'S1'])
def test_mpi3d_forecasts_replay_every_condition_and_freeze(role):
    from iwm_replication.mpi3d_byol import build_model
    from iwm_replication.campaign_mpi3d_evaluation import evaluate_matched_forecasts
    from iwm_replication.mpi3d_byol_evaluation import distribution_metrics
    from iwm_replication.campaign_evaluation import fit_readout_candidates, frozen_evaluation
    cfg = matched_config(role)
    cfg['eval']['quantiles'] = 3
    model = build_model(cfg).eval().requires_grad_(False)
    generator = torch.Generator().manual_seed(41)
    source = torch.rand(4, 3, 64, 64, generator=generator)
    target = torch.rand(4, 3, 64, 64, generator=generator)
    with torch.no_grad():
        x = model.encode(torch.cat((source, target)), branch='target')['projector'].numpy()
    y = np.arange(16).reshape(8, 2).astype(float)
    head, _ = fit_readout_candidates(x[:4], y[:4], x[4:], y[4:], [.1])
    action = torch.tensor([[-1., 0.], [1., 0.], [0., 1.], [0., -1.]])
    factors = torch.zeros(4, 7, dtype=torch.long)
    factors[:, 0] = torch.arange(4)
    factors[:, 5:7] = torch.tensor([[5,5], [6,6], [7,7], [8,8]])
    truth = torch.stack((factors[:, 5:7], factors[:, 5:7] + action), 1)
    batch = dict(x_source=source, action=action, source_factors=factors, query_index=torch.arange(4),
        target_views=torch.stack((source, target), 1), candidate_execution_succeeded=torch.tensor([[False,True]]).repeat(4,1),
        candidate_target_positions=truth)
    with frozen_evaluation(model):
        report, records, arrays = evaluate_matched_forecasts(model, [batch], cfg, torch.device('cpu'), head)
    assert report['posterior_used_for_forecasts'] is False
    for name in ('prior', 'fixed_prior_mean', 'persistence', 'wrong_action', 'wrong_source', 'encode_decode_oracle',
                 'physical_oracle', 'coordinate_persistence', 'command_success'):
        independently_scored = distribution_metrics(torch.from_numpy(arrays[name + '_physical']),
            torch.from_numpy(arrays[name + '_weights']), torch.from_numpy(arrays['truth_physical']),
            torch.from_numpy(arrays['truth_weights']), gap_epsilon=cfg['eval']['gap_epsilon'])
        np.testing.assert_allclose(records[f'physical/{name}/energy_score_euclidean'],
                                   independently_scored['energy_score_rms'].numpy() * np.sqrt(2), rtol=0, atol=0)
    np.testing.assert_allclose(records['physical/physical_oracle/energy_score_euclidean'], .25)
    np.testing.assert_allclose(records['physical/coordinate_persistence/energy_score_euclidean'], .5)
    np.testing.assert_allclose(arrays['command_success_physical'], factors[:, None, 5:7].numpy() + 4 * action[:, None].numpy())
    np.testing.assert_allclose(arrays['command_success_physical'], factors[:, None, 5:7].numpy() + 4 * action[:, None].numpy())
    assert arrays['prior_raw_projector'].shape == (4, 3 if role.endswith('1') else 1, 8)


@pytest.mark.parametrize('role', ['R0', 'R1', 'S0', 'S1'])
def test_mpi3d_full_development_and_final_artifact_reuse(tmp_path, monkeypatch, role):
    """Checkpoint loader is isolated; real model/probe/scorer paths use tiny synthetic images."""
    import iwm_replication.campaign_mpi3d_evaluation as campaign
    from iwm_replication.mpi3d_byol import build_model, resolve_matched_config
    model = build_model(matched_config(role)).eval().requires_grad_(False)
    # Production config validity and final-bank gates are exercised independently
    # of the tiny model architecture; CLI load_endpoint validates their binding.
    base = yaml.safe_load((ROOT / 'config/campaigns/five_seed_v1/mpi3d.yaml').read_text())
    cfg = resolve_matched_config(base, role=role, replication=1)
    archive = tmp_path / 'synthetic-archive'
    archive.write_bytes(b'no real MPI3D images')
    cfg['data'].update(images_path=str(archive), images_sha256=file_hash(archive))
    calls = []
    def clean_dataset(configuration, split):
        calls.append(split)
        generator = torch.Generator().manual_seed({'train': 11, 'validation': 12, 'test': 13}[split])
        probe_context = SeedContext(dataset='mpi3d', purpose='evaluation', replication=None)
        rank = np.argsort(probe_context.numpy_rng('evaluation-bank', module='online-probes', partition=split).permutation(16))
        return [dict(image=torch.rand(3,64,64,generator=generator), position=torch.tensor([i+4, i+5]),
                    camera_height=torch.tensor(int(rank[i])%3), shape_id=torch.tensor(int(rank[i])%6), color_id=torch.tensor((int(rank[i])+1)%6),
                    size_id=torch.tensor(int(rank[i])%2)) for i in range(16)]
    monkeypatch.setattr(campaign, 'build_clean_labeled_dataset', clean_dataset)
    def readouts(model, cfg, settings, context, device):
        dataset = clean_dataset(cfg, 'train')
        with torch.no_grad():
            x = model.encode(torch.stack([row['image'] for row in dataset]), branch='target')['projector'].numpy()
        y = np.stack([row['position'].numpy() for row in dataset])
        return {name: (x[a:b], y[a:b]) for name,a,b in [('fit',0,8),('development',8,12),('holdout',12,16)]}, \
               {'fit': list(range(8)), 'development': list(range(8,12)), 'holdout': list(range(12,16))}
    monkeypatch.setattr(campaign, 'readout_banks', readouts)
    def forecasts(cfg, settings, context, mode):
        dataset = clean_dataset(cfg, 'validation' if mode == 'development' else 'test')
        source = torch.stack([row['image'] for row in dataset[:4]])
        target = torch.stack([row['image'] for row in dataset[4:8]])
        action = torch.tensor([[-1.,0.],[1.,0.],[0.,1.],[0.,-1.]])
        factors = torch.zeros(4,7,dtype=torch.long)
        factors[:,0] = torch.arange(4)
        factors[:,5:7] = torch.tensor([[10,10],[12,12],[14,14],[16,16]])
        batch = dict(x_source=source, action=action, source_factors=factors, query_index=torch.arange(4),
            target_views=torch.stack((source,target),1), candidate_execution_succeeded=torch.tensor([[False,True]]).repeat(4,1),
            candidate_target_positions=torch.stack((factors[:,5:7], factors[:,5:7]+4*action),1))
        return [batch], {'canonical_split': mode, 'indices': list(range(4))}
    monkeypatch.setattr(campaign, 'campaign_forecast_loader', forecasts)
    device = torch.device('cpu')
    kwargs = dict(model=model, training_config=cfg, checkpoint=tmp_path/'fake-checkpoint.pt',
                  checkpoint_sha256='c'*64, device=device)
    before = state_hash(model)
    dev = tmp_path/'development'
    report = campaign.run(config(), output_dir=dev, **kwargs)
    assert report['frozen_state_unchanged'] and state_hash(model) == before
    assert set(calls) == {'train', 'validation'}
    fitted = dev/'fitted.pt'
    analysis = yaml.safe_load((ROOT/'config/campaigns/five_seed_v1/analysis.yaml').read_text())
    context = SeedContext(dataset='mpi3d', purpose='evaluation', replication=None)
    final_manifest = campaign.query_regeneration_manifest('mpi3d', context, config()['mpi3d'], 'final',
        {'images_sha256': cfg['data']['images_sha256'],
         'position_manifest_sha256': file_hash(cfg['data']['position_manifest_paths']['test']),
         'condition': 'balanced-S', 'population': 'canonical-MPI3D-attribute-split-v1'})
    analysis.update(frozen=True, protocol_sha256='a'*64, evaluation_contract_sha256=content_hash(config()),
                    final_bank_sha256={'moving_mnist':'b'*64, 'mpi3d':content_hash(final_manifest)})
    calls.clear()
    final = tmp_path/'final'
    campaign.run(config(), output_dir=final, mode='final', fitted_artifacts=fitted,
        fitted_sha256=file_hash(fitted), frozen_protocol_sha256='a'*64, analysis_contract=analysis, **kwargs)
    assert set(calls) == {'test'} and state_hash(model) == before
    assert not (final/'fitted.pt').exists()
    assert not (final/'probe_bank_fit.pt').exists()
    summary = json.loads((final/'summary.json').read_text())
    assert summary['partition'] == 'final' and summary['purpose'] == 'main-training'
    assert summary['evaluation_contract_sha256'] == content_hash(config())
    assert np.isfinite(summary['metrics']['physical_forecast_energy_score'])


def test_mpi3d_query_group_order_supports_wrong_source_control(monkeypatch):
    import iwm_replication.campaign_mpi3d_evaluation as campaign
    from types import SimpleNamespace
    class Queries:
        def __len__(self):
            return 32
        def __getitem__(self, index):
            return {'source_factors': torch.tensor([index//4,0,0,0,0,5,5]), 'action_id': index%4}
    monkeypatch.setattr(campaign, 'forecast_loader', lambda *a: SimpleNamespace(dataset=Queries()))
    cfg = matched_config('S1')
    cfg['eval']['batch_size'] = 4
    context = SeedContext(dataset='mpi3d', purpose='evaluation', replication=None)
    loader, manifest = campaign.campaign_forecast_loader(cfg, config()['mpi3d'], context, mode='development')
    assert len(manifest['indices']) == 8
    assert len(set(index//4 for index in manifest['indices'])) == 2
    from iwm_replication.adassl_controls import deranged_source_permutation
    for batch in loader:
        perm = deranged_source_permutation(batch['source_factors'])
        assert not (batch['source_factors'][perm] == batch['source_factors']).all(-1).any()
