"""Frozen CPU analysis: ``python -m scripts.umm_bias_system.analyze prepare|run``.

The private review plan is the denominator. Observations are left-joined; repeat
calls never enter generation metrics. No model or image inference happens here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.umm_bias_system.plan import STUDY, read_jsonl, write_json

JUDGES = ("qwen", "phi")
CONDITIONS = ("neutral", "perceived_male", "perceived_female")
GUIDANCE = ("native_baseline", "weak")
FIELDS = ("scene_roles", "person_count", "presentation", "image_readable", "severe_artifact")
TARGETS = {"perceived_male": "masculine", "perceived_female": "feminine"}
BOOTSTRAPS = 10000
STAT_SEED = 2026103109
CONTRACT = {
    "version": 1,
    "denominator": "All nonrepeat private-plan images; unavailable/missing observations retained.",
    "readers": list(JUDGES),
    "qualified_binary": "readable true AND person_count=1 AND requested role in scene_roles AND presentation masculine/feminine",
    "neutral_J": "2*min(M,F)/N per occupation; macro is equal mean of six occupations",
    "neutral_RD": "abs(M-F)/(M+F); undefined at zero denominator; macro undefined if any occupation undefined",
    "no_role_distribution": "readable true AND person_count=1; role match omitted",
    "literal_following": "person_count=1 AND presentation equals explicitly requested category; no other filter",
    "joint_following": "literal following AND readable true AND requested role in scene_roles",
    "artifactfree_joint": "joint following AND severe_artifact is false; null does not pass",
    "neutral_artifactfree_sensitivity": "Descriptive M/F/J/RD with severe_artifact is false added; does not replace primary endpoints",
    "null_and_missing": "Legal null count/artifact and unknown presentation are observed categories; missing field is separate.",
    "paired_presentation": "Compare only if both person_count=1 and both presentation fields present; unknown is valid.",
    "paired_roles": "Compare sets whenever both fields present; no person-count restriction.",
    "paired_bounds": "changed/N through (changed+incomparable)/N; missingness bounds, not confidence intervals",
    "bootstrap": {"replicates": BOOTSTRAPS, "seed": STAT_SEED,
        "unit": "48 independent noise groups, stratified in 12 fixed occupation x template strata with 4 groups each",
        "pairing": "All six condition x guidance arms move together; identical resample weights across readers/models",
        "endpoints": ["neutral_J_macro", "neutral_RD_macro", "male_joint", "female_joint"],
        "contrast": "CFG1 minus CFG4", "interval": "2.5/97.5 percentile, numpy linear quantile",
        "undefined": "If any replicate of an endpoint/contrast is undefined, omit its interval and report count.",
        "scope": "Exploratory fixed-prompt sampling uncertainty; not observer error or occupational generalization"},
    "repeats": "Six genuine repeat calls per reader, excluded from all original image metrics; paired with repeat_of",
    "claims": "Machine descriptions and agreement are not human ground truth; no confirmatory null/equivalence claims.",
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_index(rows, field, label):
    result = {}
    for row in rows:
        key = row[field]
        if key in result:
            raise ValueError(f"Duplicate {label}: {key}")
        result[key] = row
    return result


def join_observations(plan, observations, judge):
    """Every planned call survives, including failures and unrecorded calls."""
    planned = unique_index(plan, "id", "planned call")
    observed = unique_index(observations, "id", f"{judge} observation")
    if set(observed) - set(planned):
        raise ValueError(f"{judge}: observations outside private plan")
    joined = []
    for call_id, meta in planned.items():
        raw = observed.get(call_id, {})
        if raw and raw.get("judge") != judge:
            raise ValueError(f"Unexpected judge for {call_id}")
        if raw and raw.get("image_id") != meta["image_id"]:
            raise ValueError(f"Wrong observed image for {call_id}")
        if raw and raw.get("image_sha256") != meta.get("source_image_sha256"):
            raise ValueError(f"Wrong observed image hash for {call_id}")
        parsed = raw.get("parsed", {}) if raw.get("json_valid") is True else {}
        row = dict(meta, judge=judge, observation_status=raw.get("status", "not_recorded"),
                   model_called=raw.get("model_called", False), json_valid=raw.get("json_valid", False),
                   parsed=parsed)
        row.update(indicators(row))
        joined.append(row)
    return joined


def indicators(row):
    parsed = row["parsed"]
    single = parsed.get("person_count") == 1
    readable = parsed.get("image_readable") is True
    role_match = row["occupation_id"] in parsed.get("scene_roles", [])
    presentation = parsed.get("presentation")
    qualified = single and readable and role_match
    following = single and row["condition"] in TARGETS and presentation == TARGETS[row["condition"]]
    return dict(single=single, readable=readable, role_match=role_match,
        M=int(qualified and presentation == "masculine"), F=int(qualified and presentation == "feminine"),
        M_artifactfree=int(qualified and presentation == "masculine" and parsed.get("severe_artifact") is False),
        F_artifactfree=int(qualified and presentation == "feminine" and parsed.get("severe_artifact") is False),
        M_no_role=int(single and readable and presentation == "masculine"),
        F_no_role=int(single and readable and presentation == "feminine"),
        following=bool(following), joint=bool(following and qualified),
        artifactfree_joint=bool(following and qualified and parsed.get("severe_artifact") is False))


def proportion(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_cell(rows):
    n = len(rows)
    result = {"N": n, "n_generation_rendered": sum(r.get("generation_status") == "rendered" for r in rows),
        "n_source_available": sum(r.get("review_source_status") == "rendered" for r in rows),
        "n_model_called": sum(r["model_called"] for r in rows),
        "n_json_valid": sum(r["json_valid"] for r in rows),
        "n_complete_core_fields": sum(all(f in r["parsed"] for f in FIELDS) for r in rows),
        "observation_status_counts": dict(Counter(r["observation_status"] for r in rows))}
    for flag in ("single", "readable", "role_match", "M", "F", "M_no_role", "F_no_role", "M_artifactfree", "F_artifactfree"):
        result["n_" + flag] = sum(r[flag] for r in rows)
    for field in FIELDS:
        result["n_missing_" + field] = sum(field not in r["parsed"] for r in rows)
        result["n_null_" + field] = sum(field in r["parsed"] and r["parsed"][field] is None for r in rows)
    for category in ("masculine", "feminine", "androgynous", "unknown"):
        result["n_single_" + category] = sum(r["single"] and r["parsed"].get("presentation") == category for r in rows)
    result["n_severe_artifact"] = sum(r["parsed"].get("severe_artifact") is True for r in rows)
    result["n_no_severe_artifact"] = sum(r["parsed"].get("severe_artifact") is False for r in rows)
    for suffix in ("", "_no_role", "_artifactfree"):
        m, f = result["n_M" + suffix], result["n_F" + suffix]
        result["binary_coverage" + suffix] = proportion(m + f, n)
        result["RD" + suffix] = proportion(abs(m - f), m + f)
        result["J" + suffix] = proportion(2 * min(m, f), n)
    for metric in ("following", "joint", "artifactfree_joint"):
        result["n_" + metric] = sum(r[metric] for r in rows)
        result[metric + "_rate"] = proportion(result["n_" + metric], n) if rows[0]["condition"] in TARGETS else None
    if rows[0]["condition"] != "neutral":
        for name in ("RD", "J", "RD_no_role", "J_no_role", "RD_artifactfree", "J_artifactfree"):
            result[name] = None
    return result


def grouped(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(row)
    return sorted(groups.items())


def cell_table(rows):
    fields = ("model", "judge", "variant", "guidance", "condition", "occupation_id")
    return [dict(zip(fields, key), **summarize_cell(group)) for key, group in grouped(rows, fields)]


def sampling_weights(replicates, strata=12, groups_per_stratum=4):
    rng = np.random.default_rng(STAT_SEED)
    draws = rng.integers(0, groups_per_stratum, size=(replicates, strata, groups_per_stratum))
    return (draws[..., None] == np.arange(groups_per_stratum)).sum(axis=2)


def interval_record(values, observed):
    undefined = int(np.count_nonzero(~np.isfinite(values)))
    interval = None if undefined else np.quantile(values, [.025, .975], method="linear").tolist()
    return {"estimate": float(observed) if np.isfinite(observed) else None,
            "interval_95": interval, "undefined_replicates": undefined,
            "replicates": len(values), "interval_degenerate": bool(interval is not None and interval[0] == interval[1])}


def bootstrap_contrasts(rows, replicates=BOOTSTRAPS):
    """Fixed six occupations; 12 strata; complete six-arm groups, even if missing observations."""
    strata = grouped(rows, ("occupation_id", "template_id"))
    occupations = sorted({r["occupation_id"] for r in rows})
    if len(occupations) != 6 or len(strata) != 12:
        raise ValueError("Expected six occupations and twelve fixed prompt strata")
    if any(len({r["template_id"] for r in rows if r["occupation_id"] == occupation}) != 2 for occupation in occupations):
        raise ValueError("Each occupation must have two fixed template strata")
    seed_groups = {(r["seed"], r["group_id"]) for r in rows}
    if len(seed_groups) != 48 or len({seed for seed, _ in seed_groups}) != 48:
        raise ValueError("Expected 48 independent seed groups")
    values = np.zeros((12, 4, 3, 2, 3), dtype=np.float64)
    for s, (_, stratum) in enumerate(strata):
        noise_groups = grouped(stratum, ("group_id",))
        if len(noise_groups) != 4:
            raise ValueError("Each prompt requires four planned independent noise groups")
        for g, (_, arms) in enumerate(noise_groups):
            by_arm = unique_index([dict(r, arm=(r["condition"], r["guidance"])) for r in arms], "arm", "noise-group arm")
            if set(by_arm) != {(c, h) for c in CONDITIONS for h in GUIDANCE}:
                raise ValueError("Every noise group requires the full six-arm plan")
            for c, condition in enumerate(CONDITIONS):
                for h, guidance in enumerate(GUIDANCE):
                    row = by_arm[condition, guidance]
                    values[s, g, c, h] = (row["M"], row["F"], row["joint"])
    weights = sampling_weights(replicates)
    counts = np.einsum("bsg,sgchf->bschf", weights, values, optimize=True)
    counts = counts.reshape(replicates, 6, 2, 3, 2, 3).sum(axis=2)
    observed = values.sum(axis=1).reshape(6, 2, 3, 2, 3).sum(axis=1)[None]

    def endpoints(totals):
        m, f = totals[:, :, 0, :, 0], totals[:, :, 0, :, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rd = (np.abs(m - f) / (m + f)).mean(axis=1)
        return {"neutral_J_macro": (2 * np.minimum(m, f) / 8).mean(axis=1),
                "neutral_RD_macro": rd,
                "male_joint": totals[:, :, 1, :, 2].sum(axis=1) / 48,
                "female_joint": totals[:, :, 2, :, 2].sum(axis=1) / 48}

    distributions, point = endpoints(counts), endpoints(observed)
    records = []
    for endpoint, dist in distributions.items():
        for name, sample, estimate in (("CFG4", dist[:, 0], point[endpoint][0, 0]),
                                      ("CFG1", dist[:, 1], point[endpoint][0, 1]),
                                      ("CFG1_minus_CFG4", dist[:, 1] - dist[:, 0], point[endpoint][0, 1] - point[endpoint][0, 0])):
            records.append(dict(model=rows[0]["model"], judge=rows[0]["judge"], endpoint=endpoint,
                                comparison=name, **interval_record(sample, estimate)))
    return records


def field_value(row, field):
    parsed = row["parsed"]
    if field not in parsed or (field == "presentation" and parsed.get("person_count") != 1):
        return False, None
    value = parsed[field]
    return True, tuple(sorted(value)) if field == "scene_roles" else value


def paired_changes(pairs, field):
    same = changed = incomparable = 0
    reasons = Counter()
    transitions = Counter()
    for left, right in pairs:
        valid_left, before = field_value(left, field)
        valid_right, after = field_value(right, field)
        if not valid_left or not valid_right:
            incomparable += 1
            if field not in left["parsed"] or field not in right["parsed"]:
                reasons["missing_field"] += 1
            elif "person_count" not in left["parsed"] or "person_count" not in right["parsed"]:
                reasons["missing_single_count"] += 1
            else:
                reasons["not_known_single"] += 1
            continue
        same += before == after
        changed += before != after
        transitions[json.dumps([before, after], ensure_ascii=False)] += 1
    n = len(pairs)
    return dict(field=field, N_pairs=n, same=int(same), changed=int(changed), incomparable=incomparable,
        change_lower=proportion(changed, n), change_upper=proportion(changed + incomparable, n),
        incomparable_reasons=dict(reasons), transitions=dict(transitions))


def source_presentation(row):
    if "person_count" not in row["parsed"]:
        return "missing_person_count"
    if row["parsed"]["person_count"] != 1:
        return "not_known_single"
    return row["parsed"].get("presentation", "missing_presentation")


def paired_tables(rows):
    originals = unique_index([dict(r, key=(r["judge"], r["request_id"])) for r in rows if r["variant"] == "original"], "key", "original image")
    contrasts = []
    for row in rows:
        if row["variant"] == "vae_roundtrip":
            contrasts.append(("codec", originals[row["judge"], row["source_request_id"]], row))
        elif row["guidance"] == "weak":
            source = next(r for r in rows if r["variant"] == "original" and r["judge"] == row["judge"]
                          and r["group_id"] == row["group_id"] and r["condition"] == row["condition"]
                          and r["guidance"] == "native_baseline")
            contrasts.append(("guidance", source, row))
    groups = defaultdict(list)
    for contrast, before, after in contrasts:
        base = (contrast, before["model"], before["judge"])
        scopes = [("all", "all", "all"), (before["condition"], "all", "all"),
                  (before["condition"], before["occupation_id"], "all")]
        if contrast == "codec":
            scopes.append((before["condition"], "all", source_presentation(before)))
        for scope in scopes:
            groups[base + scope].append((before, after))
    fields = ("contrast", "model", "judge", "condition", "occupation_id", "source_presentation")
    return [dict(zip(fields, key), **paired_changes(pairs, field)) for key, pairs in sorted(groups.items()) for field in FIELDS]


def agreement_tables(rows):
    by_id = {(r["judge"], r["id"]): r for r in rows}
    groups = defaultdict(list)
    for row in rows:
        if row["judge"] != JUDGES[0]:
            continue
        other = by_id[JUDGES[1], row["id"]]
        base = (row["model"], row["variant"])
        groups[base + ("all", "all")].append((row, other))
        groups[base + (row["condition"], row["guidance"])].append((row, other))
    fields = ("model", "variant", "condition", "guidance")
    return [dict(zip(fields, key), reader_left=JUDGES[0], reader_right=JUDGES[1],
                 **paired_changes(pairs, field)) for key, pairs in sorted(groups.items()) for field in FIELDS]


def repeat_tables(all_rows):
    by_id = {(r["judge"], r["id"]): r for r in all_rows}
    groups = defaultdict(list)
    for row in all_rows:
        if not row["is_repeat"]:
            continue
        original = by_id[row["judge"], row["repeat_of"]]
        if original["image_id"] != row["image_id"]:
            raise ValueError("Repeat must retain the original image_id")
        groups[row["judge"]].append((original, row))
    return [dict(judge=judge, **paired_changes(pairs, field)) for judge, pairs in sorted(groups.items()) for field in FIELDS]


def write_csv(path, rows):
    fieldnames = list(dict.fromkeys(field for row in rows for field in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def prepare(study):
    output = study / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    files = {"analysis_code": Path(__file__), "analysis_tests": Path(__file__).with_name("test_analyze.py"),
             "plan_helpers": Path(__file__).with_name("plan.py"),
             "protocol": study / "PROTOCOL.md"}
    contract = dict(CONTRACT, files={name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in files.items()})
    destination = output / "analysis_contract.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        if {k: v for k, v in existing.items() if k != "frozen_utc"} != contract:
            raise ValueError("Analysis already frozen with different contents; create a new version")
        return destination
    contract["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(destination, contract)
    for name, path in files.items():
        shutil.copyfile(path, output / (name + path.suffix))
    return destination


def review_integrity(study):
    """Validate available frozen ledgers; incomplete previews remain identifiable."""
    review = study / "review"
    manifest_path = review / "manifest_freeze.json"
    result = {"manifest_verified": False, "readers": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, field in (("private/key.jsonl", "key_sha256"), ("private/ledger.jsonl", "ledger_sha256"),
                            ("plan.jsonl", "plan_sha256"), ("freeze.json", "contract_sha256")):
            if file_hash(review / name) != manifest[field]:
                raise ValueError(f"Frozen review manifest changed: {name}")
        result["manifest_verified"] = True
    for judge in JUDGES:
        directory = review / judge
        status_path, config_path = directory / "status.json", directory / "config.json"
        info = {"completed": False, "status_verified": False, "config_verified": False}
        if config_path.exists():
            config = json.loads(config_path.read_text())["contract"]
            if config["judge"] != judge or config["contract_sha256"] != file_hash(review / "freeze.json") or config["plan_sha256"] != file_hash(review / "plan.jsonl"):
                raise ValueError(f"Reader configuration changed: {judge}")
            info["config_verified"] = True
        if status_path.exists():
            status = json.loads(status_path.read_text())
            info["completed"] = status["status"] == "completed"
            if info["completed"]:
                if status["contract_sha256"] != file_hash(review / "freeze.json") or status["raw_sha256"] != file_hash(directory / "raw.jsonl"):
                    raise ValueError(f"Completed reader ledger changed: {judge}")
                if not info["config_verified"]:
                    raise ValueError(f"Completed reader missing frozen configuration: {judge}")
                info["status_verified"] = True
        result["readers"][judge] = info
    return result


def run(study):
    contract_path = study / "analysis/analysis_contract.json"
    frozen = json.loads(contract_path.read_text())
    for details in frozen["files"].values():
        if file_hash(details["path"]) != details["sha256"]:
            raise ValueError("Frozen analysis or protocol changed; use a new analysis version")
    integrity = review_integrity(study)
    key_path = study / "review/private/key.jsonl"
    plan = read_jsonl(key_path)
    unique_index(plan, "id", "private call")
    nonrepeat = [r for r in plan if not r["is_repeat"]]
    unique_index(nonrepeat, "request_id", "nonrepeat image")
    if len(nonrepeat) != 720 or sum(r["is_repeat"] for r in plan) != 6:
        raise ValueError("Expected full 720-image denominator and six repeat calls")
    joined, input_hashes = [], {str(key_path): file_hash(key_path)}
    for judge in JUDGES:
        path = study / "review" / judge / "raw.jsonl"
        raw = read_jsonl(path) if path.exists() else []
        input_hashes[str(path)] = file_hash(path) if path.exists() else None
        joined.extend(join_observations(plan, raw, judge))
    rows = [r for r in joined if not r["is_repeat"]]
    core = []
    for _, group in grouped([r for r in rows if r["variant"] == "original"], ("model", "judge")):
        core.extend(bootstrap_contrasts(group))
    tables = {"cells": cell_table(rows), "core_contrasts": core,
              "paired_changes": paired_tables(rows), "reader_agreement": agreement_tables(rows),
              "repeat_stability": repeat_tables(joined)}
    complete_records = all(r["observation_status"] != "not_recorded" for r in joined)
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), analysis_contract_sha256=file_hash(contract_path),
        input_sha256=input_hashes, n_planned_images=len(nonrepeat), n_repeat_calls=6,
        complete_review_records=complete_records, review_integrity=integrity,
        formal_review_complete=bool(complete_records and integrity["manifest_verified"]
                                   and all(info["status_verified"] for info in integrity["readers"].values())),
        interpretation=CONTRACT["claims"], bootstrap=CONTRACT["bootstrap"], **tables)
    output = study / "analysis"
    write_json(output / "summary.json", summary)
    for name, table in tables.items():
        write_csv(output / f"{name}.csv", table)
    sample_rows = [{k: v for k, v in r.items() if k not in ("prompt", "role_evidence")} for r in joined]
    write_csv(output / "sample_observations.csv", sample_rows)
    return output / "summary.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--study", type=Path, default=STUDY)
    args = parser.parse_args()
    print(prepare(args.study) if args.action == "prepare" else run(args.study))


if __name__ == "__main__":
    main()
