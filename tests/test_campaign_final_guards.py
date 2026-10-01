"""Final Moving-MNIST gates must reject bad provenance before image/label access."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml

from src import campaign_moving_mnist_evaluation as evaluation
from src.campaign_evaluation import query_regeneration_manifest, save_fitted
from src.moving_mnist import GeneratorConfig
from src.seed_streams import SeedContext, content_sha256


ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT/'config/campaigns/five_seed_v1/evaluation.yaml').read_text())
    ctx = SeedContext('moving_mnist', 'main-training', 1)
    fixed = SeedContext('moving_mnist', 'evaluation', None)
    training = {'purpose': 'five-seed-campaign-training', 'campaign_recipe': {'fixture': True},
                'seed_context': ctx.as_dict(),
                'train_data': {'identity_manifest_sha256': 'e'*64, 'generator': asdict(GeneratorConfig())}}
    model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    checkpoint = tmp_path/'checkpoint.pt'
    torch.save({'model_config': {'role': 'R0'}, 'training_config': {'total_steps': 75000}}, checkpoint)
    # Model/checkpoint correctness has dedicated actual-model integration tests.
    # Here isolate ordering of gates relative to every dataset access.
    monkeypatch.setattr(evaluation, 'load_frozen', lambda *args, **kwargs: (model, training))
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid final provenance reached identity manifest or final data')
    monkeypatch.setattr(evaluation, 'load_identity_manifest', forbidden)
    monkeypatch.setattr(evaluation, 'load_digits', forbidden)
    monkeypatch.setattr(evaluation, 'make_campaign_banks', forbidden)
    query = query_regeneration_manifest('moving_mnist', fixed, config['moving_mnist'], 'final',
        {'identity_manifest_sha256': training['train_data']['identity_manifest_sha256'],
         'generator': training['train_data']['generator']})
    analysis = yaml.safe_load((ROOT/'config/campaigns/five_seed_v1/analysis.yaml').read_text())
    analysis.update(frozen=True, protocol_sha256='a'*64, evaluation_contract_sha256=content_sha256(config),
                    final_bank_sha256={'moving_mnist': content_sha256(query), 'mpi3d': 'b'*64})
    fitted = {'version': config['version'], 'checkpoint_sha256': 'c'*64,
              'configuration_sha256': content_sha256(config), 'role': 'R0', 'readout': {}, 'probes': {}}
    return config, training, analysis, fitted, {
        'checkpoint': checkpoint, 'expected_sha256': 'c'*64, 'data_dir': tmp_path/'data',
        'output_dir': tmp_path/'output', 'device': torch.device('cpu'), 'mode': 'final',
        'fitted_artifacts': tmp_path/'fitted.pt', 'frozen_protocol_sha256': 'a'*64,
    }


@pytest.mark.parametrize('change', ['missing', 'draft', 'protocol', 'evaluation', 'bank'])
def test_frozen_binding_failure_never_opens_final_data(tmp_path, monkeypatch, change):
    config, _, analysis, fitted, kwargs = fixture(tmp_path, monkeypatch)
    kwargs['fitted_sha256'] = save_fitted(kwargs['fitted_artifacts'], fitted)
    if change == 'draft':
        analysis['frozen'] = False
    elif change == 'protocol':
        analysis['protocol_sha256'] = 'd'*64
    elif change == 'evaluation':
        analysis['evaluation_contract_sha256'] = 'd'*64
    elif change == 'bank':
        analysis['final_bank_sha256']['moving_mnist'] = 'd'*64
    with pytest.raises(ValueError):
        evaluation.run(config, analysis_contract=None if change == 'missing' else analysis, **kwargs)
    assert not kwargs['output_dir'].exists()


@pytest.mark.parametrize('change', ['missing', 'checksum', 'checkpoint_sha256', 'configuration_sha256', 'role', 'version'])
def test_fitted_artifact_failure_never_opens_final_data(tmp_path, monkeypatch, change):
    config, _, analysis, fitted, kwargs = fixture(tmp_path, monkeypatch)
    if change in ('checkpoint_sha256', 'configuration_sha256'):
        fitted[change] = 'd'*64
    elif change in ('role', 'version'):
        fitted[change] = 'different'
    kwargs['fitted_sha256'] = 'd'*64 if change == 'missing' else save_fitted(kwargs['fitted_artifacts'], fitted)
    if change == 'checksum':
        kwargs['fitted_sha256'] = 'd'*64
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluation.run(config, analysis_contract=analysis, **kwargs)
    assert not kwargs['output_dir'].exists()


@pytest.mark.parametrize('change', ['legacy_purpose', 'missing_campaign_recipe', 'development_checkpoint'])
def test_nonmain_or_noncampaign_checkpoint_cannot_open_final_data(tmp_path, monkeypatch, change):
    config, training, analysis, fitted, kwargs = fixture(tmp_path, monkeypatch)
    kwargs['fitted_sha256'] = save_fitted(kwargs['fitted_artifacts'], fitted)
    if change == 'legacy_purpose':
        training['purpose'] = 'training-integration-test-only'
    elif change == 'missing_campaign_recipe':
        del training['campaign_recipe']
    else:
        training['seed_context'] = SeedContext('moving_mnist', 'development-training', 0).as_dict()
    with pytest.raises(ValueError):
        evaluation.run(config, analysis_contract=analysis, **kwargs)
    assert not kwargs['output_dir'].exists()


def test_changed_identity_manifest_rejected_before_final_image_bank(tmp_path, monkeypatch):
    config, _, analysis, fitted, kwargs = fixture(tmp_path, monkeypatch)
    kwargs['fitted_sha256'] = save_fitted(kwargs['fitted_artifacts'], fitted)
    monkeypatch.setattr(evaluation, 'load_identity_manifest', lambda path: {'sha256': 'f'*64})
    with pytest.raises(ValueError, match='Identity manifest'):
        evaluation.run(config, analysis_contract=analysis, **kwargs)
