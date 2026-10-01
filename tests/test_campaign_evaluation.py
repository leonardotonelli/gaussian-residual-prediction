"""Independent score arithmetic, leakage guards, and frozen readout contracts."""
import hashlib
import random

import numpy as np
import pytest
import torch

from src.campaign_evaluation import (
    assert_disjoint_partitions, feature_transform, fit_affine_mse, fit_readout_candidates,
    frozen_evaluation, iid_scores, load_fitted, require_final_contract, save_fitted,
    split_development_indices,
)
from src.moving_mnist_evaluation import predict_velocity
from src.moving_mnist_metrics import physical_scores
from src.mpi3d_byol_evaluation import distribution_metrics
from src.seed_streams import SeedContext


def test_analytic_two_atom_iid_and_weighted_estimators_are_distinct():
    atoms = np.array([[[-1.], [1.]]])
    iid = iid_scores(atoms, atoms)
    weighted = distribution_metrics(torch.from_numpy(atoms), torch.full((1, 2), .5),
                                    torch.from_numpy(atoms), torch.full((1, 2), .5), gap_epsilon=1e-8)
    assert iid['energy_score'][0] == 0.  # 1 - 4/(2*2*1)
    assert weighted['energy_score_rms'][0].item() == .5  # 1 - .5*(.5*2)
    point = iid_scores(np.array([[[0.]]]), atoms)
    assert point['energy_score'][0] == 1.
    assert point['coverage_squared'][0] == 1.


def test_scores_match_independent_loops_including_weighted_zero_mass_and_chunking():
    rng = np.random.default_rng(75)
    predictions = rng.normal(size=(3, 5, 2))
    truths = rng.normal(size=(3, 7, 2))
    expected = []
    coverage = []
    for p, y in zip(predictions, truths):
        cross = sum(float(np.linalg.norm(a-b)) for a in p for b in y) / (len(p)*len(y))
        self_term = sum(float(np.linalg.norm(p[i]-p[j])) for i in range(len(p)) for j in range(len(p)) if i != j) / (2*len(p)*(len(p)-1))
        expected.append(cross - self_term)
        coverage.append(sum(min(float(np.linalg.norm(a-b)) ** 2 for a in p) for b in y) / len(y))
    actual = iid_scores(predictions, truths)
    np.testing.assert_allclose(actual['energy_score'], expected, atol=1e-12)
    np.testing.assert_allclose(actual['coverage_squared'], coverage, atol=1e-12)
    chunked = physical_scores(torch.from_numpy(predictions), torch.from_numpy(truths), chunk_size=2)
    np.testing.assert_allclose(chunked['energy_score'], expected, atol=1e-12)
    weights, truth_weights = np.array([[0., .1, .2, .3, .4]] * 3), np.array([[.2, .8]] * 3)
    exact = distribution_metrics(torch.from_numpy(predictions), torch.from_numpy(weights),
                                 torch.from_numpy(truths[:, :2]), torch.from_numpy(truth_weights), gap_epsilon=1e-8)
    expected_weighted = []
    for p, y, w, q in zip(predictions, truths[:, :2], weights, truth_weights):
        cross = sum(w[i]*q[j]*np.sqrt(np.mean((p[i]-y[j])**2)) for i in range(5) for j in range(2))
        self_term = .5*sum(w[i]*w[j]*np.sqrt(np.mean((p[i]-p[j])**2)) for i in range(5) for j in range(5))
        expected_weighted.append(cross-self_term)
    np.testing.assert_allclose(exact['energy_score_rms'], expected_weighted, atol=1e-12)


def test_readout_selects_only_real_selection_features_and_fits_normalizer_on_fit():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(30, 3)) * rng.uniform(.1, 20, size=(30, 1))
    selection = rng.normal(size=(17, 3)) * rng.uniform(.1, 20, size=(17, 1))
    coeff = np.array([[2., -1.], [.3, 1.5], [-.4, 2.]])
    y = feature_transform(x, 'unit') @ coeff
    sy = feature_transform(selection, 'unit') @ coeff
    head, details = fit_readout_candidates(x, y, selection, sy, [1e-8, .01, 1.])
    assert details['selected_transform'] == 'unit'
    np.testing.assert_allclose(head['mean'], feature_transform(x, 'unit').mean(0))
    # Forecast inputs use the same transform; positive radial rescaling cannot change a unit readout.
    np.testing.assert_allclose(predict_velocity(selection, head), predict_velocity(selection * 10, head), atol=1e-12)
    np.testing.assert_allclose(predict_velocity(selection, head), sy, atol=1e-6)
    assert not details['forecast_scores_used_for_selection']
    # Report data is deliberately absent from the fitting API; arbitrarily extreme held-out rows do not mutate it.
    before = head['mean'].copy()
    predict_velocity(np.full((4, 3), 1e6), head)
    np.testing.assert_array_equal(before, head['mean'])
    tie, tied_details = fit_readout_candidates(x, np.zeros((30, 2)), selection, np.zeros((17, 2)), [.1])
    assert tied_details['selected_transform'] == tie['feature_transform'] == 'raw'


def test_rank_deficient_affine_probe_solves_mse_without_inventing_rank():
    x = np.arange(10., dtype=np.float64)
    features = np.column_stack((x, 2*x, np.ones(10)))
    targets = np.column_stack((3*x+2, -x+4))
    head, details = fit_affine_mse(features, targets)
    assert details['rank_deficient'] and details['rank'] == 1
    assert head['active'].tolist() == [True, True, False]
    np.testing.assert_allclose(predict_velocity(features, head), targets, atol=1e-12)
    assert details['normalizer_fit_partition'] == 'fit only'


def test_disjoint_fixed_development_split_and_final_access_guard():
    context = SeedContext(dataset='mpi3d', purpose='evaluation', replication=None)
    partitions = split_development_indices(11, seed_context=context, module='probe-bank')
    replay = split_development_indices(11, seed_context=context, module='probe-bank')
    assert_disjoint_partitions(partitions)
    assert set(np.concatenate(list(partitions.values()))) == set(range(11))
    for name in partitions:
        np.testing.assert_array_equal(partitions[name], replay[name])
    with pytest.raises(ValueError, match='Overlapping'):
        assert_disjoint_partitions({'fit': [1, 2], 'selection': [2, 3]})
    with pytest.raises(ValueError):
        require_final_contract('final')
    require_final_contract('development')
    require_final_contract('final', fitted_artifacts='fitted.pt', fitted_sha256='a'*64,
                           frozen_protocol_sha256='b'*64)
    with pytest.raises(ValueError):
        require_final_contract('final', fitted_artifacts='fitted.pt', fitted_sha256='bad',
                               frozen_protocol_sha256='b'*64)


def test_fitted_artifacts_round_trip_hash_and_no_overwrite(tmp_path):
    path = tmp_path / 'fitted.pt'
    payload = {'head': {'weight': np.arange(6.).reshape(3, 2), 'feature_transform': 'unit'}, 'config': {'v': 1}}
    digest = save_fitted(path, payload)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    restored = load_fitted(path, digest)
    np.testing.assert_array_equal(restored['head']['weight'], payload['head']['weight'])
    assert restored['head']['feature_transform'] == 'unit'
    with pytest.raises(FileExistsError):
        save_fitted(path, payload)
    path.write_bytes(path.read_bytes() + b'tampering')
    with pytest.raises(ValueError, match='SHA256'):
        load_fitted(path, digest)


def test_frozen_guard_preserves_rng_and_detects_bn_buffer_mutation():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3)).eval().requires_grad_(False)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with frozen_evaluation(model):
        model(torch.randn(4, 3))
        random.random()
        np.random.rand(4)
    assert random.getstate() == py_state
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    with pytest.raises(RuntimeError, match='mutated frozen'):
        with frozen_evaluation(model):
            model[1].running_mean.add_(1)
    model.train()
    with pytest.raises(ValueError, match='eval mode'):
        with frozen_evaluation(model):
            pass
