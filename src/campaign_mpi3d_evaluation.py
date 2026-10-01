"""Matched four-role MPI3D frozen evaluation with disjoint selection/report banks."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .adassl_controls import deranged_source_permutation
from .campaign_evaluation import (EVALUATION_VERSION, fit_readout_candidates, frozen_evaluation,
    load_fitted, require_final_contract, save_fitted, write_json, validate_evaluation_config,
    query_regeneration_manifest, validate_final_bindings, runtime_metadata)
from .data import build_clean_labeled_dataset
from .mpi3d_data import MPI3DArchive, MPI3D_ACTION_STRIDE
from .mpi3d_byol_evaluation import (distribution_metrics, evaluation_seed_context, evaluate_probes,
    feature_diagnostics, fit_probes, forecast_loader, summarize_records)
from .mpi3d_pilot_inspection import InspectionImages, image_partitions
from .mpi3d_probes import MPI3D_PROBE_LABEL_KEYS
from .seed_streams import content_sha256 as content_hash
from .moving_mnist_evaluation import file_hash, predict_velocity, regression_metrics, state_hash, transform


class IndexedQueries(Dataset):
    def __init__(self, dataset, ids):
        self.dataset, self.ids = dataset, list(map(int, ids))
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        return {**self.dataset[self.ids[index]], 'query_index': self.ids[index]}


def _loader(dataset, cfg, context, partition):
    return DataLoader(dataset, batch_size=cfg['eval']['batch_size'], shuffle=False,
        num_workers=cfg['data']['num_workers'], generator=context.torch_generator('loader',
        module='campaign-evaluation', partition=partition))


@torch.inference_mode()
def probe_banks(model, cfg, settings, context, device, *, mode):
    banks, ids = {}, {}
    names = ('fit', 'selection', 'report') if mode == 'development' else ('report',)
    datasets = {}
    for name in names:
        split = 'train' if name == 'fit' else 'validation' if mode == 'development' else 'test'
        if split not in datasets:
            datasets[split] = build_clean_labeled_dataset(cfg, split)
        dataset = datasets[split]
        order = context.numpy_rng('evaluation-bank', module='online-probes', partition=split).permutation(len(dataset))
        count = settings['probe_counts'][name]
        start = settings['probe_counts']['selection'] if name == 'report' and mode == 'development' else 0
        selected = order[start:start + count]
        if len(selected) != count:
            raise ValueError('Probe bank request exceeds distinct image population')
        rows = {key: [] for key in ('pooled', 'projector', *MPI3D_PROBE_LABEL_KEYS)}
        for batch in _loader(Subset(dataset, selected.tolist()), cfg, context, 'probe-' + name):
            encoded = model.encode(batch['image'].to(device), branch='online')
            for key in ('pooled', 'projector'):
                rows[key].append(encoded[key].cpu())
            for key in MPI3D_PROBE_LABEL_KEYS:
                rows[key].append(batch[key].cpu())
        banks[name] = {'features': {key: torch.cat(rows[key]) for key in ('pooled', 'projector')},
                       'labels': {key: torch.cat(rows[key]) for key in MPI3D_PROBE_LABEL_KEYS}}
        ids[name] = {'canonical_split': split, 'indices': selected.tolist()}
    return banks, ids


@torch.inference_mode()
def readout_banks(model, cfg, settings, context, device):
    # Fixed train-attribute image identities; all positions. Selection/report are disjoint.
    partitions = image_partitions(seed=context.seed('evaluation-bank', module='physical-readout'),
                                  counts=tuple(settings['readout_counts']))
    archive = MPI3DArchive(cfg['data']['images_path'])
    bank = {}
    for name, ids in partitions.items():
        features, positions = [], []
        for batch in _loader(InspectionImages(archive, ids), cfg, context, 'readout-' + name):
            features.append(model.encode(batch['image'].to(device), branch='target')['projector'].cpu().numpy())
            positions.append(batch['position'].numpy())
        bank[name] = (np.concatenate(features), np.concatenate(positions))
    return bank, {name: ids.tolist() for name, ids in partitions.items()}


def campaign_forecast_loader(cfg, settings, context, *, mode):
    split = 'validation' if mode == 'development' else 'test'
    dataset = forecast_loader(cfg, split).dataset
    if len(dataset) % 4 or settings['queries'] % 4:
        raise ValueError('Balanced command evaluation requires four actions per selected source')
    # Sampling groups retains each source's four actions; all roles see the same bank.
    order = context.numpy_rng('evaluation-bank', module='forecast-source-groups', partition=split).permutation(len(dataset) // 4)
    groups = order[:settings['queries'] // 4]
    ids = (groups[None] * 4 + np.arange(4)[:, None]).reshape(-1)
    if len(ids) != settings['queries']:
        raise ValueError('Forecast query request exceeds source population')
    return _loader(IndexedQueries(dataset, ids), cfg, context, 'forecast-' + split), {'canonical_split': split, 'indices': ids.tolist()}


@torch.inference_mode()
def evaluate_matched_forecasts(model, loader, cfg, device, readout):
    records, arrays = {}, {}
    def append(destination, name, value):
        destination.setdefault(name, []).append(value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value))
    def score(name, predictions, weights, targets, truth):
        values = distribution_metrics(predictions, weights, targets, truth, gap_epsilon=cfg['eval']['gap_epsilon'])
        if name.startswith('physical/'):
            # Two coordinates: RMS distance * sqrt(2) is Euclidean grid distance.
            values['energy_score_euclidean'] = values['energy_score_rms'] * np.sqrt(2.)
        for key, value in values.items():
            append(records, name + '/' + key, value)
    for batch in loader:
        source, action = batch['x_source'].to(device), batch['action'].to(device)
        z = model.encode(source, branch='online')['projector']
        quantiles = cfg['eval']['quantiles']
        prior, weights = model.forecast_features(z, action, quantiles=quantiles, normalize=False)
        fixed, single = model.forecast_features(z, action, fixed_residual=True, normalize=False)
        wrong_action = torch.stack((-action[:, 1], action[:, 0]), -1)
        shuffled, shuffled_weights = model.forecast_features(z, wrong_action, quantiles=quantiles, normalize=False)
        permutation = deranged_source_permutation(batch['source_factors']).to(device)
        wrong_source, source_weights = model.forecast_features(z[permutation], action, quantiles=quantiles, normalize=False)
        persistence = model.encode(source, branch='target')['projector'][:, None]
        # Future images and simulator labels are first accessed after all forecasts.
        images = batch['target_views'].to(device)
        if not torch.equal(batch['candidate_execution_succeeded'].cpu(), torch.tensor([[False, True]]).expand(len(source), -1)):
            raise ValueError('Balanced S truth must enumerate failure then success')
        targets = torch.stack([model.encode(images[:, i], branch='target')['projector'] for i in range(2)], 1)
        truth = prior.new_full((len(source), 2), .5)
        conditions = {'prior': (prior, weights), 'fixed_prior_mean': (fixed, single),
            'persistence': (persistence, single), 'wrong_action': (shuffled, shuffled_weights),
            'wrong_source': (wrong_source, source_weights), 'encode_decode_oracle': (targets, truth)}
        physical_truth = batch['candidate_target_positions'].to(device).double()
        for name, (p, w) in conditions.items():
            normalized = F.normalize(p, dim=-1)
            score('latent/' + name, normalized, w, F.normalize(targets, dim=-1), truth)
            decoded = torch.from_numpy(predict_velocity(p.flatten(0, 1).cpu().numpy(), readout)).to(device).reshape(len(source), -1, 2)
            score('physical/' + name, decoded, w, physical_truth, truth)
            append(arrays, name + '_raw_projector', p)
            append(arrays, name + '_weights', w)
            append(arrays, name + '_physical', decoded)
        physical_source = batch['source_factors'][:, 5:7].to(device).double()[:, None]
        baselines = {'physical_oracle': (physical_truth, truth), 'coordinate_persistence': (physical_source, single),
                     'command_success': (physical_source + MPI3D_ACTION_STRIDE * action[:, None], single)}
        for name, (p, w) in baselines.items():
            score('physical/' + name, p, w, physical_truth, truth)
            append(arrays, name + '_physical', p)
            append(arrays, name + '_weights', w)
        for name, value in (('source_factors', batch['source_factors']), ('action', action), ('query_index', batch['query_index']),
                            ('truth_physical', physical_truth), ('truth_weights', truth), ('truth_raw_projector', targets)):
            append(arrays, name, value)
        if model.is_stochastic:
            pm, ps = model.prior(torch.cat((z, action), -1))
            append(arrays, 'prior_mean', pm)
            append(arrays, 'prior_std', ps.exp())
            for outcome in range(2):
                future_z = model.encode(images[:, outcome], branch='online')['projector']
                qm, qs = model.posterior(torch.cat((z, action, future_z), -1))
                kl = (ps - qs + .5 * ((qs - ps).mul(2).exp() + (qm - pm).square() * (-2 * ps).exp() - 1)).sum(-1)
                append(arrays, f'diagnostic_posterior_mean_{outcome}', qm)
                append(arrays, f'diagnostic_posterior_std_{outcome}', qs.exp())
                append(records, f'diagnostic_posterior_kl_{outcome}', kl)
    if not records:
        raise ValueError('Empty forecast bank')
    records, arrays = ({key: np.concatenate(values) for key, values in collection.items()} for collection in (records, arrays))
    report = {'num_queries': len(arrays['action']), 'forecast_atoms': arrays['prior_weights'].shape[1],
        'estimator': 'full weighted finite-support Energy Score; fixed Gaussian midpoint quadrature, not IID U statistic',
        'physical_primary_units': 'Euclidean grid-index distance; datasets analyzed separately',
        'latent_space': 'unit-normalized actual target projector; EMA R / shared S',
        'metrics': summarize_records(records), 'truth': 'exact balanced failure/success distribution',
        'posterior_used_for_forecasts': False,
        'real_target_readout': regression_metrics(arrays['encode_decode_oracle_physical'].reshape(-1, 2), arrays['truth_physical'].reshape(-1, 2)),
        'target_features': feature_diagnostics(torch.from_numpy(arrays['truth_raw_projector'].reshape(-1, arrays['truth_raw_projector'].shape[-1]))),
        'predicted_feature_support': {}}
    for name in ('prior', 'persistence', 'encode_decode_oracle'):
        raw = arrays[name + '_raw_projector'].reshape(-1, arrays[name + '_raw_projector'].shape[-1])
        norms = np.linalg.norm(transform(raw, readout), axis=-1)
        report['predicted_feature_support'][name] = {'standardized_norm_p99': float(np.quantile(norms, .99)),
            'fraction_above_fit_p99': float(np.mean(norms > readout['train_standardized_norm_p99']))}
    report['warning'] = 'Readout pipeline score; oracle distortion is diagnostic and is never subtracted. Latent scores are not a cross-model physical ranking.'
    return report, records, arrays


def run(config, *, model, training_config, checkpoint, checkpoint_sha256, output_dir, device,
        mode='development', fitted_artifacts=None, fitted_sha256=None, frozen_protocol_sha256=None, analysis_contract=None):
    require_final_contract(mode, fitted_artifacts=fitted_artifacts, fitted_sha256=fitted_sha256,
                           frozen_protocol_sha256=frozen_protocol_sha256)
    validate_evaluation_config(config)
    from .mpi3d_byol import validate_config
    validate_config(training_config)
    settings, cfg = config['mpi3d'], deepcopy(training_config)
    context = evaluation_seed_context(cfg)
    if context is None or not hasattr(model, 'target_branch'):
        raise ValueError('Campaign evaluation requires matched four-role model and fixed seed context')
    if mode == 'final' and cfg['stage'] != 'main-training':
        raise ValueError('Final evaluation accepts only main training replications')
    cfg['eval'].update(batch_size=settings['batch_size'], quantiles=settings['quantiles'])
    cfg['probe'] = settings['probe']
    split_name = 'validation' if mode == 'development' else 'test'
    query_manifest = query_regeneration_manifest('mpi3d', context, settings, mode,
        {'images_sha256': cfg['data']['images_sha256'],
         'position_manifest_sha256': file_hash(cfg['data']['position_manifest_paths'][split_name]),
         'condition': 'balanced-S', 'population': 'canonical-MPI3D-attribute-split-v1'})
    query_hash = content_hash(query_manifest)
    validate_final_bindings(mode, config, 'mpi3d', query_manifest, analysis_contract=analysis_contract,
                            frozen_protocol_sha256=frozen_protocol_sha256)
    if mode == 'final':
        fitted = load_fitted(fitted_artifacts, fitted_sha256)
        if (fitted['version'] != EVALUATION_VERSION or fitted['checkpoint_sha256'] != checkpoint_sha256
                or fitted['configuration_sha256'] != content_hash(config) or fitted['role'] != cfg['role']):
            raise ValueError('Fitted artifact does not match checkpoint/configuration')
        def tensors(value):
            if isinstance(value, dict):
                return {k: tensors(v) for k, v in value.items()}
            return torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        probe_heads, readout = tensors(fitted['probes']), fitted['readout']
    if file_hash(cfg['data']['images_path']) != cfg['data']['images_sha256']:
        raise ValueError('MPI3D image archive checksum mismatch')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with frozen_evaluation(model) as before:
        banks, probe_ids = probe_banks(model, cfg, settings, context, device, mode=mode)
        for name, bank in banks.items():
            torch.save(bank, output_dir / f'probe_bank_{name}.pt')
        if mode == 'development':
            probe_heads, probe_selection = fit_probes(banks['fit'], banks['selection'], cfg, device)
            readout_data, readout_ids = readout_banks(model, cfg, settings, context, device)
            readout, selection = fit_readout_candidates(*readout_data['fit'], *readout_data['development'],
                settings['ridge_alphas'], transforms=settings['readout_transforms'])
            readout_report = regression_metrics(predict_velocity(readout_data['holdout'][0], readout), readout_data['holdout'][1])
            fitted = {'version': EVALUATION_VERSION, 'checkpoint_sha256': checkpoint_sha256, 'role': cfg['role'],
                'configuration_sha256': content_hash(config), 'probes': probe_heads, 'readout': readout,
                'probe_selection': probe_selection, 'readout_selection': selection, 'readout_report': readout_report,
                'readout_image_ids': readout_ids, 'probe_ids': probe_ids}
            fitted_sha256 = save_fitted(output_dir / 'fitted.pt', fitted)
            for name, (x, y) in readout_data.items():
                np.savez_compressed(output_dir / f'readout_{name}.npz', features=x, position=y)
        probes = evaluate_probes(banks['report'], probe_heads, device, seed_context=context)
        loader, query_bank = campaign_forecast_loader(cfg, settings, context, mode=mode)
        forecasts, scores, arrays = evaluate_matched_forecasts(model, loader, cfg, device, readout)
        realized_query_hash = content_hash({'bank': query_bank, 'source_factors': arrays['source_factors'].tolist(),
            'actions': arrays['action'].tolist(), 'truth_positions': arrays['truth_physical'].tolist()})
        np.savez_compressed(output_dir / 'per_query.npz', **scores)
        np.savez_compressed(output_dir / 'forecasts.npz', **arrays)
        write_json(output_dir / 'banks.json', {'query_regeneration_manifest': query_manifest, 'realized_query_sha256': realized_query_hash, 'probes': probe_ids, 'queries': query_bank,
            'readout_training_image_ids': fitted['readout_image_ids']})
    repo = Path(__file__).resolve().parents[1]
    files = [Path(module.__file__).resolve() for name, module in sys.modules.items()
             if name.startswith('src') and getattr(module, '__file__', '').endswith('.py')]
    files.extend([repo / 'scripts/evaluate_five_seed_mpi3d.py', repo / 'scripts/evaluate_mpi3d_byol.py'])
    code = {str(path.relative_to(repo)): file_hash(path) for path in files}
    contract = {'version': EVALUATION_VERSION, 'configuration': config, 'partition': mode, 'role': cfg['role'], 'runtime': runtime_metadata(device),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha256, 'readout_sha256': fitted_sha256,
        'query_bank_sha256': query_hash, 'realized_query_bank_sha256': realized_query_hash, 'seed_context': context.as_dict(),
        'frozen_protocol_sha256': frozen_protocol_sha256, 'training_config_sha256': content_hash(training_config),
        'test_set_accessed': mode == 'final', 'code_sha256': code,
        'split_history': 'Existing MPI3D attribute/position split was historically inspected; fresh seeds do not erase exposure'}
    write_json(output_dir / 'contract.json', contract)
    report = {'status': 'complete', 'probes': probes, 'forecasts': forecasts,
        'readout_selection': fitted['readout_selection'], 'readout_in_support_report': fitted['readout_report'],
        'state_sha256_before': before, 'state_sha256_after': state_hash(model), 'frozen_state_unchanged': True}
    write_json(output_dir / 'metrics.json', report)
    write_json(output_dir / 'summary.json', {'schema': 'five-seed-model-result-v1', 'campaign': context.campaign,
        'protocol_sha256': frozen_protocol_sha256 or content_hash(config), 'dataset': 'mpi3d', 'role': cfg['role'],
        'replication': cfg['seed'], 'purpose': cfg['stage'], 'partition': mode, 'status': 'complete', 'head_restart': 0,
        'query_bank_sha256': query_hash, 'evaluation_contract_sha256': content_hash(config), 'evaluation_run_sha256': content_hash(contract),
        'checkpoint_sha256': checkpoint_sha256, 'readout_sha256': fitted_sha256,
        'source_sha256': content_hash(code), 'config_sha256': content_hash(config),
        'metrics': {'physical_forecast_energy_score': float(scores['physical/prior/energy_score_euclidean'].mean())}})
    return report
