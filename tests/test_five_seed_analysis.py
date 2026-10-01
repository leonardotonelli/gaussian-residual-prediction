"""The analysis unit is a trained seed, never a query or probe restart."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from iwm_replication.five_seed_analysis import (
    analyze_campaign, exact_sign_flip_pvalue, holm_adjust, markdown_report,
    validate_analysis_contract,
)
from iwm_replication.five_seed_campaign import run_matrix
from iwm_replication.seed_streams import content_sha256


def frozen_contract():
    contract = yaml.safe_load(Path('config/campaigns/five_seed_v1/analysis.yaml').read_text())
    contract.update(frozen=True, protocol_sha256='a' * 64, evaluation_contract_sha256='b' * 64,
                    final_bank_sha256={'moving_mnist': 'c' * 64, 'mpi3d': 'd' * 64})
    return contract


def complete_rows(contract):
    rows = []
    offsets = {'R0': 4., 'R1': 3., 'S0': 2., 'S1': 1.}
    for entry in run_matrix():
        for restart in contract['head_restarts']:
            row = {name: entry[name] for name in ('dataset', 'role', 'replication', 'purpose')}
            row.update(schema='five-seed-model-result-v1', campaign=contract['campaign'],
                       partition='final', status='complete', head_restart=restart,
                       protocol_sha256=contract['protocol_sha256'],
                       evaluation_contract_sha256=contract['evaluation_contract_sha256'],
                       query_bank_sha256=contract['final_bank_sha256'][row['dataset']],
                       checkpoint_sha256=content_sha256(entry), readout_sha256='f' * 64,
                       source_sha256='1' * 64, config_sha256='2' * 64,
                       metrics={'physical_forecast_energy_score': offsets[row['role']] + row['replication'] + restart})
            rows.append(row)
    return rows


def test_complete_paired_effects_and_interval_use_five_differences():
    contract = frozen_contract()
    rows = complete_rows(contract)
    # S1-R1 differences in Moving-MNIST become exactly 1,2,3,4,5.
    for row in rows:
        if row['dataset'] == 'moving_mnist' and row['role'] == 'S1':
            row['metrics']['physical_forecast_energy_score'] = 3. + 2 * row['replication']
    report = analyze_campaign(rows, contract)
    assert report['counts'] == {'complete': 40, 'failed': 0, 'missing': 0, 'incomplete-heads': 0}
    result = report['datasets']['moving_mnist']['metrics']['physical_forecast_energy_score']['contrasts']['S1-R1']
    assert result['paired_differences'] == [1., 2., 3., 4., 5.]
    assert result['n'] == 5 and result['mean'] == 3
    assert result['sample_sd'] == pytest.approx(2.5 ** .5)
    # Independent tabulated df=4 critical value and hand-computed standard error sqrt(.5).
    assert result['interval_95'] == pytest.approx([3 - 2.7764451051977987 * .5 ** .5,
                                                 3 + 2.7764451051977987 * .5 ** .5])
    assert 'sign_flip_pvalue' not in result
    json.dumps(report, allow_nan=False)
    assert '40/40' in markdown_report(report)


def test_head_restarts_average_within_five_models_and_missing_heads_stay_visible():
    contract = frozen_contract()
    contract['head_restarts'] = [0, 1, 2]
    rows = complete_rows(contract)
    report = analyze_campaign(rows, contract)
    role = report['datasets']['mpi3d']['metrics']['physical_forecast_energy_score']['roles']['S1']
    assert role['n'] == 5 and role['values'] == [3., 4., 5., 6., 7.]
    rows.pop()  # MPI3D S1 seed5, restart2
    partial = analyze_campaign(rows, contract)
    assert partial['counts']['incomplete-heads'] == 1
    contrast = partial['datasets']['mpi3d']['metrics']['physical_forecast_energy_score']['contrasts']['S1-R1']
    assert contrast['n'] == 4 and contrast['interval_95'] is None
    assert contrast['paired_differences'][-1] is None


def test_failed_missing_and_empty_matrix_account_for_all_40_entries():
    contract = frozen_contract()
    rows = complete_rows(contract)
    failed = rows.pop(0)
    failed.update(status='failed', reason='Nonfinite objective; original run retained', metrics={})
    rows.append(failed)
    rows.pop(1)
    report = analyze_campaign(rows, contract)
    assert len(report['registry']) == 40 and not report['complete']
    assert report['counts'] == {'complete': 38, 'failed': 1, 'missing': 1, 'incomplete-heads': 0}
    empty = analyze_campaign([], contract)
    assert empty['counts']['missing'] == 40
    assert empty['datasets']['mpi3d']['metrics']['physical_forecast_energy_score']['roles']['R0']['mean'] is None
    json.dumps(empty, allow_nan=False)


@pytest.mark.parametrize('change', [
    {'purpose': 'development-training'}, {'partition': 'selection'}, {'replication': 6},
    {'replication': True}, {'protocol_sha256': '0' * 64}, {'query_bank_sha256': '0' * 64},
    {'evaluation_contract_sha256': '0' * 64}, {'head_restart': 1}, {'readout_sha256': 'missing'},
    {'metrics': {'physical_forecast_energy_score': float('nan')}},
])
def test_reject_mixed_or_unverified_input(change):
    contract = frozen_contract()
    rows = complete_rows(contract)
    rows[0].update(change)
    with pytest.raises(ValueError):
        analyze_campaign(rows, contract)


def test_no_duplicate_or_substituted_checkpoints_across_head_restarts():
    contract = frozen_contract()
    rows = complete_rows(contract)
    with pytest.raises(ValueError, match='Duplicate'):
        analyze_campaign(rows + [deepcopy(rows[0])], contract)
    rows[1]['checkpoint_sha256'] = rows[0]['checkpoint_sha256']
    with pytest.raises(ValueError, match='different trained replications'):
        analyze_campaign(rows, contract)
    contract['head_restarts'] = [0, 1]
    rows = complete_rows(contract)
    rows[1]['checkpoint_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='same trained model'):
        analyze_campaign(rows, contract)


def test_draft_contract_cannot_score_main_results():
    contract = frozen_contract()
    contract['frozen'] = False
    validate_analysis_contract(contract, require_frozen=False)
    with pytest.raises(ValueError, match='frozen contract'):
        analyze_campaign([], contract)


def test_exact_test_resolution_zero_ties_symmetry_and_holm_family():
    assert exact_sign_flip_pvalue([1, 2, 3, 4, 5]) == 2 / 32
    assert exact_sign_flip_pvalue([-1, -2, -3, -4, -5]) == 2 / 32
    assert exact_sign_flip_pvalue([0, 0, 0, 0, 0]) == 1
    assert exact_sign_flip_pvalue([1, -1, 2, -2, 0]) == 1
    assert holm_adjust([.03, .01, .04]) == pytest.approx([.06, .03, .06])
    contract = frozen_contract()
    contract.update(hypothesis_tests='two-sided-exact-sign-flip',
                    multiplicity='holm-across-all-declared-contrasts-and-datasets')
    report = analyze_campaign(complete_rows(contract), contract)
    result = report['datasets']['moving_mnist']['metrics']['physical_forecast_energy_score']['contrasts']['S1-R1']
    assert result['sign_flip_pvalue'] == .0625
    assert result['holm_pvalue'] == .25  # Four prespecified dataset/contrast tests.
    contract['multiplicity'] = 'not-applicable'
    with pytest.raises(ValueError, match='multiplicity'):
        validate_analysis_contract(contract)


def test_summary_cli_writes_replayable_json_and_markdown_without_overwrite(tmp_path, monkeypatch):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location('campaign_summary_cli', 'scripts/summarize_five_seed_campaign.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    contract = frozen_contract()
    contract_path = tmp_path / 'analysis.yaml'
    contract_path.write_text(yaml.safe_dump(contract))
    result_path = tmp_path / 'results.json'
    result_path.write_text(json.dumps(complete_rows(contract)))
    output = tmp_path / 'summary'
    monkeypatch.setattr(sys, 'argv', ['summary', '--contract', str(contract_path), '--results', str(result_path), '--output', str(output)])
    cli.main()
    payload = json.loads((output / 'analysis.json').read_text())
    assert payload['complete'] and payload['expected_training_entries'] == 40
    assert '40/40' in (output / 'summary.md').read_text()
    before = (output / 'analysis.json').read_bytes()
    with pytest.raises(FileExistsError):
        cli.main()
    assert (output / 'analysis.json').read_bytes() == before
