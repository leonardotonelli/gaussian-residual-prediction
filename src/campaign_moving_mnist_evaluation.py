"""Executable frozen Moving-MNIST campaign evaluation; no training or final selection."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

from .campaign_evaluation import (EVALUATION_VERSION, feature_transform, fit_affine_mse,
    fit_readout_candidates, frozen_evaluation, iid_scores, load_fitted, require_final_contract,
    save_fitted, write_json, validate_evaluation_config, query_regeneration_manifest, validate_final_bindings, runtime_metadata)
from .moving_mnist import GeneratorConfig, SourceState, render_centers, trajectory_centers
from .moving_mnist_data import load_digits, load_identity_manifest
from .moving_mnist_evaluation import (extract_features, file_hash, fit_digit_probe, forecast_bank,
    load_frozen, make_banks, predict_velocity, regression_metrics, state_hash, transform)
from .moving_mnist_stream import OnlineClips
from .seed_streams import SeedContext, content_sha256 as content_hash


class FinalEvaluationClips(OnlineClips):
    """Final-only adapter; legacy/training OnlineClips keeps its test-data ban.

    The shared implementation supplies pure address/render operations. Its
    temporary development label is replaced before the bank is returned; only
    explicit fixed final evaluation streams are allowed here.
    """
    def __init__(self, *args, seed_context, seed_partition, **kwargs):
        if (seed_context.purpose != 'evaluation' or seed_context.replication is not None
                or seed_partition not in ('final-report', 'final-queries')):
            raise ValueError('Final clips require a fixed final evaluation stream')
        super().__init__(*args, split='development', seed_context=seed_context,
                         seed_partition=seed_partition, **kwargs)
        self.split = 'test'
        self.contract.update(version='five-seed-frozen-final-clips-v1', split='test', training_access=False)


def make_campaign_banks(data_dir, manifest, training, settings, context, *, mode):
    if mode == 'development':
        return make_banks(data_dir, manifest, training['train_data'], settings['sizes'], settings['seed'], seed_context=context)
    if mode != 'final':
        raise ValueError('Unknown bank mode')
    raw = load_digits(data_dir, manifest, 'test')
    return {name: FinalEvaluationClips(*raw, seed=settings['seed'],
                generator_config=GeneratorConfig(**training['train_data']['generator']),
                identity_hash=manifest['sha256'], size=settings['sizes'][name], seed_context=context,
                seed_partition='final-' + name) for name in ('report', 'queries')}


def digit_metrics(x, labels, head):
    logits = transform(x, head) @ head['weight'].T + head['bias']
    shifted = logits - logits.max(1, keepdims=True)
    return {'accuracy': float(np.mean(logits.argmax(1) == labels)),
            'cross_entropy': float(np.mean(np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(labels)), labels]))}


def fit_paper_probes(features, settings, context, device):
    heads, selection = {}, {}
    for clip in ('source', 'target'):
        fit, choose = features['fit'][clip], features['selection'][clip]
        for view in ('online_backbone', 'online_projector'):
            key = clip + '/' + view
            velocity, details = fit_affine_mse(fit[view], fit['velocity'])
            digit, digit_head = fit_digit_probe(fit[view], fit['digit'], choose[view], choose['digit'],
                choose[view], choose['digit'], config=settings['digit_probe'], seed=settings['seed'],
                device=device, seed_context=context, probe_name=key)
            heads[key] = {'velocity': velocity, 'digit': digit_head}
            selection[key] = {'velocity': details, 'digit_selection': digit,
                              'primary_population': clip == 'source'}
    return heads, selection


def report_paper_probes(features, heads):
    result = {}
    for name, head in heads.items():
        clip, view = name.split('/')
        bank = features[clip]
        result[name] = {'velocity': regression_metrics(predict_velocity(bank[view], head['velocity']), bank['velocity']),
                        'digit': digit_metrics(bank[view], bank['digit'], head['digit']),
                        'mean_coordinate_variance': float(bank[view].var(0).mean())}
    return result


@torch.inference_mode()
def latent_forecasts(model, dataset, physical_arrays, *, samples, latent_futures, batch_size, context, device):
    """Replay prior stream and encode independent physical truth/oracle draws.

    Targets/surrogates are used only after every forecast for that query; Gaussian
    posterior diagnostics are marked as training-information diagnostics.
    """
    arrays = {}
    def append(name, value):
        arrays.setdefault(name, []).append(np.asarray(value))
    def encoded(image, state, velocities):
        parts, blank, cropped = [], [], []
        for start in range(0, len(velocities), batch_size):
            clips = torch.stack([render_centers(image, trajectory_centers(state, velocity)[3:6], dataset.config).unsqueeze(0)
                                 for velocity in velocities[start:start + batch_size]])
            blank.extend((clips.sum((-1, -2)) == 0).cpu().numpy().reshape(-1).tolist())
            centers = np.stack([trajectory_centers(state, v)[3:6] for v in velocities[start:start + batch_size]])
            half = dataset.config.digit_size / 2
            cropped.extend(((centers - half < -.5) | (centers + half > dataset.config.canvas_size - .5)).any(-1).reshape(-1).tolist())
            parts.append(model.encode(clips.to(device), branch='target')['projector'].cpu().numpy())
        return np.concatenate(parts), float(np.mean(blank)), float(np.mean(cropped))
    rng = context.torch_generator('evaluation-forecast', device=device, partition=dataset.seed_partition)
    stochastic = model.config.role in ('R1', 'S1')
    for query in range(len(dataset)):
        record, clips = dataset.record(query), dataset[query]
        source = clips['source'][None].to(device)
        kwargs = {'num_samples': samples, 'generator': rng} if stochastic else {}
        prior = model.forecast(source, **kwargs)
        append('prior', prior['prediction'][0].cpu().numpy())
        if stochastic:
            append('fixed_prior_mean', model.forecast(source, fixed_residual=True)['prediction'][0].cpu().numpy())
        append('persistence', model.encode(source, branch='target')['projector'].cpu().numpy())
        state = SourceState(**record['state'])
        image = dataset.images[record['image_index']]
        oracle_velocity = physical_arrays['prediction_independent_privileged_oracle'][query]
        for name, velocities in (('oracle', oracle_velocity), ('truth', physical_arrays['truth'][query, :latent_futures])):
            real, blank, cropped = encoded(image, state, velocities)
            append(name, real)
            append(name + '_blank_frame_fraction', np.asarray(blank))
            append(name + '_geometric_crop_frame_fraction', np.asarray(cropped))
        if stochastic:
            z = model.encode(source, branch='online')['projector']
            u = model.encode(clips['surrogate'][None].to(device), branch='online')['projector']
            qm, qs = model.posterior(torch.cat((z, u), -1))
            for name, tensor in (('prior_mean', prior['prior_mean']), ('prior_std', prior['prior_std']),
                                 ('posterior_mean', qm), ('posterior_std', qs)):
                append(name, tensor[0].cpu().numpy())
        append('source_blank_frame_fraction', np.asarray((source.sum((-1, -2)) == 0).float().mean().cpu()))
        append('query_index', np.asarray(query))
    arrays = {name: np.stack(rows) for name, rows in arrays.items()}
    truth = feature_transform(arrays['truth'], 'unit')
    scores = {name: iid_scores(feature_transform(arrays[name], 'unit'), truth)
              for name in ('prior', 'persistence', 'oracle', 'fixed_prior_mean') if name in arrays}
    spread = np.sqrt(truth.var(1).sum(-1))
    report = {'space': 'unit-normalized actual target projector; EMA R / shared S',
        'estimator': 'IID off-diagonal U-statistic; deterministic singleton correction zero',
        'latent_truth_draws': truth.shape[1], 'near_zero_truth_spread_queries': int((spread <= 1e-8).sum()),
        'metrics': {name: {key: float(values.mean()) for key, values in score.items()} for name, score in scores.items()},
        'warning': 'Separately learned latent spaces do not define a common physical model ranking'}
    if stochastic:
        pm, ps, qm, qs = (arrays[k] for k in ('prior_mean', 'prior_std', 'posterior_mean', 'posterior_std'))
        kl = (np.log(ps / qs) + .5 * ((qs / ps) ** 2 + ((qm - pm) / ps) ** 2 - 1)).sum(-1)
        report['gaussian_diagnostics'] = {'mean_training_information_kl': float(kl.mean()),
            'prior_std_mean': float(ps.mean()), 'posterior_std_mean': float(qs.mean()),
            'posterior_used_for_forecasts': False}
        arrays['diagnostic_posterior_kl'] = kl
    return arrays, scores, report


def run(config, *, checkpoint, expected_sha256, data_dir, output_dir, device, mode='development',
        fitted_artifacts=None, fitted_sha256=None, frozen_protocol_sha256=None, analysis_contract=None, allow_partial_software_smoke=False):
    require_final_contract(mode, fitted_artifacts=fitted_artifacts, fitted_sha256=fitted_sha256,
                           frozen_protocol_sha256=frozen_protocol_sha256)
    validate_evaluation_config(config)
    if allow_partial_software_smoke and mode != 'development':
        raise ValueError('Partial software-smoke evaluation is development-only')
    settings = config['moving_mnist']
    if min(settings['samples'], settings['futures'], settings['latent_futures']) < 2 or settings['latent_futures'] > settings['futures']:
        raise ValueError('Invalid independent forecast/truth draw counts')
    checkpoint, output_dir, data_dir = Path(checkpoint), Path(output_dir), Path(data_dir)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    role = state['model_config']['role']
    step = state['step'] if allow_partial_software_smoke else state['training_config']['total_steps']
    del state
    repo = Path(__file__).resolve().parents[1]
    model, training = load_frozen(checkpoint, expected_sha256=expected_sha256, expected_step=step,
                                  role=role, repo=repo, device=device,
                                  **({'allow_partial_software_smoke': True} if allow_partial_software_smoke else {}))
    if training.get('purpose') != 'five-seed-campaign-training' or 'campaign_recipe' not in training:
        raise ValueError('Campaign evaluation requires the accepted campaign training recipe')
    runtime = runtime_metadata(device)
    if 'environment' in training and runtime != training['environment']:
        raise ValueError('Use the recorded training runtime for frozen campaign evaluation')
    training_context = SeedContext.from_dict(training['seed_context'])
    if mode == 'final' and training_context.purpose != 'main-training':
        raise ValueError('Only main training replications can enter final evaluation')
    context = replace(training_context, purpose='evaluation', replication=None)
    query_manifest = query_regeneration_manifest('moving_mnist', context, settings, mode,
        {'identity_manifest_sha256': training['train_data']['identity_manifest_sha256'],
         'generator': training['train_data']['generator']})
    query_hash = content_hash(query_manifest)
    validate_final_bindings(mode, config, 'moving_mnist', query_manifest, analysis_contract=analysis_contract,
                            frozen_protocol_sha256=frozen_protocol_sha256)
    if mode == 'final':
        fitted = load_fitted(fitted_artifacts, fitted_sha256)
        if (fitted['version'] != EVALUATION_VERSION or fitted['checkpoint_sha256'] != expected_sha256
                or fitted['configuration_sha256'] != content_hash(config) or fitted['role'] != role):
            raise ValueError('Frozen fitted artifact does not match this checkpoint/configuration')
        readout, probe_heads = fitted['readout'], fitted['probes']
    output_dir.mkdir(parents=True, exist_ok=False)
    with frozen_evaluation(model) as before:
        manifest = load_identity_manifest(data_dir / settings['identity_manifest'])
        if manifest['sha256'] != training['train_data']['identity_manifest_sha256']:
            raise ValueError('Identity manifest differs from the pinned training split')
        banks = make_campaign_banks(data_dir, manifest, training, settings, context, mode=mode)
        bank_records = {name: [bank.record(i) for i in range(len(bank))] for name, bank in banks.items()}
        realized_query_hash = content_hash({'records': bank_records['queries'], 'contract': banks['queries'].contract})
        features = {}
        for name in ('fit', 'selection', 'report'):
            if name not in banks:
                continue
            features[name] = {}
            for clip in ('source', 'target'):
                values = extract_features(model, banks[name], settings['batch_size'], device, clip=clip)
                features[name][clip] = values
                np.savez_compressed(output_dir / f'features_{name}_{clip}.npz', **values)
        if mode == 'development':
            probe_heads, probe_selection = fit_paper_probes(features, settings, context, device)
            fit, selection = features['fit']['target'], features['selection']['target']
            readout, readout_selection = fit_readout_candidates(fit['target_projector'], fit['velocity'],
                selection['target_projector'], selection['velocity'], settings['ridge_alphas'], transforms=settings['readout_transforms'])
            fitted = {'version': EVALUATION_VERSION, 'checkpoint_sha256': expected_sha256, 'role': role,
                'configuration_sha256': content_hash(config), 'readout': readout, 'probes': probe_heads,
                'probe_selection': probe_selection, 'readout_selection': readout_selection,
                'fit_bank_sha256': content_hash(bank_records['fit']), 'selection_bank_sha256': content_hash(bank_records['selection'])}
            fitted_sha256 = save_fitted(output_dir / 'fitted.pt', fitted)
        probes = report_paper_probes(features['report'], probe_heads)
        report = features['report']['target']
        readout_report = regression_metrics(predict_velocity(report['target_projector'], readout), report['velocity'])
        forecasts, diagnostics, arrays, per_query = forecast_bank(model, banks['queries'], readout,
            samples=settings['samples'], futures=settings['futures'], seed=settings['seed'], device=device,
            batch_size=settings['batch_size'], train_norm_p99=readout['train_standardized_norm_p99'], seed_context=context)
        latent_arrays, latent_scores, latent_report = latent_forecasts(model, banks['queries'], arrays,
            samples=settings['samples'], latent_futures=settings['latent_futures'], batch_size=settings['batch_size'],
            context=context, device=device)
        diagnostics['visibility'] = {}
        energy = np.asarray(per_query['scores']['model_prior']['energy_score'])
        distortion = np.square(arrays['prediction_encoded_decoded_oracle'] - arrays['prediction_independent_privileged_oracle']).sum(-1).mean(1)
        for name in ('truth_blank_frame_fraction', 'truth_geometric_crop_frame_fraction', 'oracle_blank_frame_fraction', 'oracle_geometric_crop_frame_fraction'):
            values = latent_arrays[name]
            strata = {}
            for label, mask in (('none', values == 0), ('any', values > 0)):
                strata[label] = {'queries': int(mask.sum()),
                    'physical_prior_energy_score': float(energy[mask].mean()) if mask.any() else None,
                    'oracle_decode_squared_error': float(distortion[mask].mean()) if mask.any() else None}
            diagnostics['visibility'][name] = {'mean_frame_fraction': float(values.mean()), 'strata': strata}
        diagnostics['visibility']['definition'] = 'Blank means zero rendered pixel mass; cropping uses nominal digit box beyond canvas; truth rates use the declared latent truth prefix'
        # Same fixed prior stream in both passes, transform applied on real/predicted features identically.
        reconstructed = np.stack([predict_velocity(row, readout) for row in latent_arrays['prior']])
        np.testing.assert_allclose(reconstructed, arrays['prediction_model_prior'], rtol=0, atol=0)
        arrays['query_index'] = np.arange(len(banks['queries']))
        arrays['query_id_sha256'] = np.asarray([content_hash(row) for row in bank_records['queries']])
        interval = np.quantile(arrays['prediction_model_prior'], [.05, .95], axis=1)
        coverage = ((arrays['truth'] >= interval[0, :, None]) & (arrays['truth'] <= interval[1, :, None])).mean(1)
        arrays['marginal_90_interval'] = interval.transpose(1, 0, 2)
        arrays['marginal_90_truth_coverage_xy'] = coverage
        diagnostics['marginal_90_interval'] = {'coverage_xy': coverage.mean(0).tolist(),
            'mean_width_xy': (interval[1] - interval[0]).mean(0).tolist(),
            'interpretation': 'Empirical marginal intervals; singleton forecasts have zero width; sample count affects coverage'}
        np.savez_compressed(output_dir / 'physical_forecasts.npz', **arrays)
        np.savez_compressed(output_dir / 'latent_forecasts.npz', **latent_arrays)
        write_json(output_dir / 'per_query.json', {'physical': per_query, 'latent': latent_scores})
        write_json(output_dir / 'banks.json', {'query_regeneration_manifest': query_manifest, 'realized_query_sha256': realized_query_hash, 'contracts': {k: b.contract for k, b in banks.items()}, 'records': bank_records})
    files = [Path(module.__file__).resolve() for name, module in sys.modules.items()
             if name.startswith('src') and getattr(module, '__file__', '').endswith('.py')]
    files.append(repo / 'scripts/evaluate_five_seed_moving_mnist.py')
    contract = {'version': EVALUATION_VERSION, 'configuration': config, 'partition': mode, 'role': role, 'runtime': runtime,
        'checkpoint_sha256': expected_sha256, 'readout_sha256': fitted_sha256, 'query_bank_sha256': query_hash,
        'realized_query_bank_sha256': realized_query_hash,
        'seed_context': context.as_dict(), 'frozen_protocol_sha256': frozen_protocol_sha256,
        'training_contract_sha256': content_hash(training), 'test_set_accessed': mode == 'final',
        'code_sha256': {str(path.relative_to(repo)): file_hash(path) for path in files}}
    write_json(output_dir / 'contract.json', contract)
    metrics = {'status': 'complete', 'probes': probes, 'forecast_readout': readout_report,
        'readout_selection': fitted['readout_selection'], 'probe_selection': fitted['probe_selection'],
        'forecasts': forecasts, 'diagnostics': diagnostics, 'latent': latent_report,
        'state_sha256_before': before, 'state_sha256_after': state_hash(model), 'frozen_state_unchanged': True}
    write_json(output_dir / 'metrics.json', metrics)
    summary = {'schema': 'five-seed-model-result-v1', 'campaign': context.campaign,
        'protocol_sha256': frozen_protocol_sha256 or content_hash(config), 'dataset': 'moving_mnist',
        'role': role, 'replication': training_context.replication, 'purpose': training_context.purpose, 'partition': mode,
        'query_bank_sha256': query_hash, 'evaluation_contract_sha256': content_hash(config), 'evaluation_run_sha256': content_hash(contract),
        'checkpoint_sha256': expected_sha256, 'readout_sha256': fitted_sha256, 'status': 'complete',
        'source_sha256': content_hash(contract['code_sha256']), 'config_sha256': content_hash(config),
        'head_restart': 0, 'metrics': {'physical_forecast_energy_score': forecasts['model_prior']['strata']['all']['energy_score']}}
    write_json(output_dir / 'summary.json', summary)
    return metrics
