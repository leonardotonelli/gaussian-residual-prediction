"""Prespecified training-replication analysis, separate from query uncertainty.

Inputs are verified per-model summaries. This module does not score images or
select a readout, endpoint, seed, contrast, or metric after seeing outcomes.
"""
from __future__ import annotations

import itertools
import math
import statistics

from .five_seed_campaign import DATASETS, MAIN_REPLICATIONS, MANIFEST_SHA256, ROLES
from .seed_streams import CAMPAIGN, content_sha256

T_975_DF4 = 2.7764451051977987
CONTRASTS = (("S1", "R1"), ("S1", "S0"))


def _digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")


def validate_analysis_contract(contract: dict, *, require_frozen=True) -> None:
    if contract.get("schema") != "five-seed-analysis-contract-v1" or contract.get("campaign") != CAMPAIGN:
        raise ValueError("Unsupported analysis contract/campaign")
    if contract.get("replications") != list(MAIN_REPLICATIONS):
        raise ValueError("Exactly five prespecified training replications are required")
    if contract.get("datasets") != list(DATASETS) or contract.get("roles") != list(ROLES):
        raise ValueError("Both datasets and four matched roles are required")
    if contract.get("contrasts") != [list(pair) for pair in CONTRASTS]:
        raise ValueError("Primary contrasts must be prespecified S1-R1 and S1-S0")
    if contract.get("interval") != {"method": "paired-student-t", "level": 0.95, "degrees_of_freedom": 4}:
        raise ValueError("Expected the declared paired 95% t interval with four degrees of freedom")
    if contract.get("seed_manifest_sha256") != MANIFEST_SHA256:
        raise ValueError("Analysis must use the accepted seed manifest")
    restarts = contract.get("head_restarts")
    if (not isinstance(restarts, list) or not restarts or any(type(v) is not int or v < 0 for v in restarts)
            or sorted(set(restarts)) != restarts):
        raise ValueError("Head restarts must be a sorted nonempty list of distinct nonnegative integers")
    metrics = contract.get("metrics_by_dataset")
    if not isinstance(metrics, dict) or set(metrics) != set(DATASETS):
        raise ValueError("Declare metrics separately for both datasets")
    for names in metrics.values():
        if (not isinstance(names, list) or not names or any(not isinstance(v, str) or not v for v in names)
                or len(set(names)) != len(names)):
            raise ValueError("Declare a nonempty distinct metric list for each dataset")
    tests = contract.get("hypothesis_tests")
    if tests not in ("none", "two-sided-exact-sign-flip"):
        raise ValueError("Only no tests or prespecified two-sided exact sign-flip tests are supported")
    expected_multiplicity = "not-applicable" if tests == "none" else "holm-across-all-declared-contrasts-and-datasets"
    if contract.get("multiplicity") != expected_multiplicity:
        raise ValueError("Prespecify multiplicity for the entire reported test family")
    if type(contract.get("frozen")) is not bool:
        raise ValueError("An explicit frozen flag is required")
    if require_frozen and not contract["frozen"]:
        raise ValueError("Main analysis requires the development-reviewed frozen contract")
    if contract["frozen"]:
        for field in ("protocol_sha256", "evaluation_contract_sha256"):
            _digest(contract.get(field), field)
        banks = contract.get("final_bank_sha256")
        if not isinstance(banks, dict) or set(banks) != set(DATASETS):
            raise ValueError("Bind a fixed final query bank for each dataset")
        for name, digest in banks.items():
            _digest(digest, name + " final bank")


def describe(values) -> dict:
    values = [float(v) for v in values]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) >= 2 else None}


def exact_sign_flip_pvalue(differences) -> float:
    """Enumerate all 32 sign assignments; zeros/ties remain in the enumeration.

    The conventional absolute-mean, two-sided test has minimum p=2/32. Its
    symmetry/exchangeability assumptions are not proved by paired sampling.
    """
    differences = [float(value) for value in differences]
    if len(differences) != 5 or not all(math.isfinite(value) for value in differences):
        raise ValueError("Exact sign-flip inference requires five finite paired differences")
    observed = abs(math.fsum(differences))
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=5):
        statistic = abs(math.fsum(sign * value for sign, value in zip(signs, differences)))
        if statistic >= observed or math.isclose(statistic, observed, rel_tol=1e-12, abs_tol=0.0):
            extreme += 1
    return extreme / 32.0


def holm_adjust(pvalues) -> list[float]:
    pvalues = list(pvalues)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in pvalues):
        raise ValueError("Expected finite probabilities")
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [0.0] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def analyze_campaign(rows: list[dict], contract: dict) -> dict:
    """Account for all 40 models; average declared head restarts within a model.

    Missing/failed inputs stay visible. A partial seed/head set cannot receive
    the full-five interval or significance test. Only rows from the frozen
    main/final pipeline are accepted; development outcomes fail validation.
    """
    validate_analysis_contract(contract)
    groups = {}
    checkpoint_owners = {}
    for row in rows:
        if row.get("schema") != "five-seed-model-result-v1" or row.get("campaign") != CAMPAIGN:
            raise ValueError("Expected campaign model-result records")
        ds, role, rep = row.get("dataset"), row.get("role"), row.get("replication")
        if ds not in DATASETS or role not in ROLES or type(rep) is not int or rep not in MAIN_REPLICATIONS:
            raise ValueError("Results must identify one of the 40 main training entries")
        if row.get("purpose") != "main-training" or row.get("partition") != "final":
            raise ValueError("Development/smoke/selection results cannot enter main analysis")
        if row.get("protocol_sha256") != contract["protocol_sha256"]:
            raise ValueError("Mixed scientific protocols are not a matched analysis")
        status = row.get("status")
        if status not in ("complete", "failed"):
            raise ValueError("Only verified complete results or explicitly failed runs may be imported")
        key = (ds, role, rep)
        group = groups.setdefault(key, [])
        if status == "failed":
            if not isinstance(row.get("reason"), str) or not row["reason"].strip() or row.get("metrics"):
                raise ValueError("A failed run needs a reason and cannot carry scored outcomes")
            if group:
                raise ValueError("A failed run cannot coexist with a scored or duplicate row")
        else:
            if any(item["status"] == "failed" for item in group):
                raise ValueError("A scored row cannot coexist with a failed run")
            restart = row.get("head_restart", 0)
            if type(restart) is not int or restart not in contract["head_restarts"]:
                raise ValueError("Undeclared readout/probe restart")
            if any(item.get("head_restart", 0) == restart for item in group):
                raise ValueError("Duplicate trained-replication/head result")
            if row.get("query_bank_sha256") != contract["final_bank_sha256"][ds]:
                raise ValueError("Query banks must match across all models and replications")
            if row.get("evaluation_contract_sha256") != contract["evaluation_contract_sha256"]:
                raise ValueError("Evaluation contract mismatch")
            for field in ("checkpoint_sha256", "readout_sha256", "source_sha256", "config_sha256"):
                _digest(row.get(field), field)
            previous_owner = checkpoint_owners.setdefault(row["checkpoint_sha256"], key)
            if previous_owner != key:
                raise ValueError("One checkpoint cannot count as different trained replications or roles")
            for item in group:
                if any(item[field] != row[field] for field in ("checkpoint_sha256", "source_sha256", "config_sha256")):
                    raise ValueError("Head restarts must belong to the same trained model and source/configuration")
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError("Complete results need a metrics mapping")
            for name in contract["metrics_by_dataset"][ds]:
                value = metrics.get(name)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"Missing or nonfinite declared metric {name}")
        group.append(row)

    registry, datasets = [], {}
    complete_models = {}
    for ds in DATASETS:
        for role in ROLES:
            for rep in MAIN_REPLICATIONS:
                key = (ds, role, rep)
                group = groups.get(key, [])
                record = {"dataset": ds, "role": role, "replication": rep}
                if not group:
                    record.update(status="missing", reason="No verified result record supplied")
                elif group[0]["status"] == "failed":
                    record.update(status="failed", reason=group[0]["reason"])
                elif {row.get("head_restart", 0) for row in group} != set(contract["head_restarts"]):
                    record.update(status="incomplete-heads", reason="Not all prespecified head restarts are available",
                                  available_head_restarts=sorted(row.get("head_restart", 0) for row in group))
                else:
                    means = {name: statistics.mean(row["metrics"][name] for row in group)
                             for name in contract["metrics_by_dataset"][ds]}
                    complete_models[key] = means
                    record.update(status="complete", metrics=means,
                                  head_restarts=list(contract["head_restarts"]),
                                  checkpoint_sha256=group[0]["checkpoint_sha256"],
                                  readout_sha256=[row["readout_sha256"] for row in sorted(group, key=lambda x: x.get("head_restart", 0))])
                registry.append(record)

    test_records = []
    for ds in DATASETS:
        dataset_report = {"metrics": {}}
        for metric in contract["metrics_by_dataset"][ds]:
            role_reports = {}
            for role in ROLES:
                values = [complete_models.get((ds, role, rep), {}).get(metric) for rep in MAIN_REPLICATIONS]
                available = [v for v in values if v is not None]
                role_reports[role] = dict(describe(available), values=values, replications=list(MAIN_REPLICATIONS),
                                         complete=len(available) == 5)
            contrast_reports = {}
            for left, right in CONTRASTS:
                differences = [None if x is None or y is None else x - y
                               for x, y in zip(role_reports[left]["values"], role_reports[right]["values"])]
                available = [v for v in differences if v is not None]
                result = dict(describe(available), paired_differences=differences,
                              replications=list(MAIN_REPLICATIONS), complete=len(available) == 5,
                              interval_95=None)
                if len(available) == 5:
                    half = T_975_DF4 * result["sample_sd"] / math.sqrt(5)
                    result["interval_95"] = [result["mean"] - half, result["mean"] + half]
                    if contract["hypothesis_tests"] != "none":
                        result["sign_flip_pvalue"] = exact_sign_flip_pvalue(available)
                contrast_reports[left + "-" + right] = result
                test_records.append(result)
            dataset_report["metrics"][metric] = {"roles": role_reports, "contrasts": contrast_reports}
        datasets[ds] = dataset_report
    if contract["hypothesis_tests"] != "none":
        # Missing tests remain in the declared family conservatively as p=1.
        adjusted = holm_adjust(record.get("sign_flip_pvalue", 1.0) for record in test_records)
        for record, value in zip(test_records, adjusted):
            record["holm_pvalue"] = value if "sign_flip_pvalue" in record else None
    counts = {status: sum(row["status"] == status for row in registry)
              for status in ("complete", "failed", "missing", "incomplete-heads")}
    return {
        "schema": "five-seed-analysis-result-v1", "campaign": CAMPAIGN,
        "analysis_contract_sha256": content_sha256(contract),
        "protocol_sha256": contract["protocol_sha256"],
        "input_sha256": content_sha256(rows), "expected_training_entries": 40,
        "complete": counts["complete"] == 40, "counts": counts,
        "registry": registry, "datasets": datasets,
        "interpretation": {
            "independent_unit": "trained replication; head restarts averaged within model",
            "uncertainty": "training randomness conditional on the fixed benchmark",
            "interval_assumption": "independent paired differences across replications and approximately normal mean inference at n=5",
            "physical_score": "full predictor/readout pipeline; no oracle subtraction",
            "sign_flip_assumption": "exchangeability under sign flips/symmetric paired differences under the null",
            "minimum_two_sided_sign_flip_p": 0.0625,
            "query_intervals": "not training-replication intervals and not produced here",
            "partial_results": "available-seed summaries are descriptive; missing pairs receive no five-seed interval/test",
        },
    }


def markdown_report(report: dict) -> str:
    lines = ["# Five-seed campaign analysis", "",
             "Verified complete entries: {complete}/40; failed: {failed}; missing: {missing}; incomplete heads: {incomplete-heads}.".format(**report["counts"]), "",
             "Unit: one trained replication. Head restarts are averaged within each model. Datasets and physical units remain separate.", ""]
    for dataset, body in report["datasets"].items():
        lines.extend(["## " + dataset, ""])
        for metric, tables in body["metrics"].items():
            lines.extend(["### " + metric, "", "| Role | Replications 1–5 | Mean | Sample SD | n |", "|---|---|---|---|---|"])
            def number(value):
                return "missing" if value is None else f"{value:.8g}"
            for role, record in tables["roles"].items():
                lines.append("| {} | {} | {} | {} | {} |".format(role, ", ".join(map(number, record["values"])), number(record["mean"]), number(record["sample_sd"]), record["n"]))
            lines.extend(["", "| Contrast | Paired differences 1–5 | Mean difference | 95% paired t interval | n |", "|---|---|---|---|---|"])
            for name, record in tables["contrasts"].items():
                interval = record["interval_95"]
                lines.append("| {} | {} | {} | {} | {} |".format(name, ", ".join(map(number, record["paired_differences"])), number(record["mean"]), "unavailable" if interval is None else "[" + ", ".join(map(number, interval)) + "]", record["n"]))
            lines.append("")
            if any("sign_flip_pvalue" in record for record in tables["contrasts"].values()):
                lines.extend(["Prespecified two-sided exact sign-flip tests (32 assignments):", ""])
                for name, record in tables["contrasts"].items():
                    lines.append(f"- {name}: raw p={number(record.get('sign_flip_pvalue'))}; Holm-adjusted p={number(record.get('holm_pvalue'))}.")
                lines.append("")
    lines.extend(["## Missing or failed entries", ""])
    failures = [row for row in report["registry"] if row["status"] != "complete"]
    lines.extend([f"- {row['dataset']} {row['role']} replication {row['replication']}: {row['status']} — {row['reason']}" for row in failures] or ["None."])
    lines.extend(["", "Intervals use four degrees of freedom only when all five paired values exist. Physical ES describes the full predictor/readout pipeline. No finding requires model superiority or statistical significance.", ""])
    return "\n".join(lines)
