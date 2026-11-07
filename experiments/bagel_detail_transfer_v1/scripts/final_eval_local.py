"""Versioned, resumable final G evaluation using the frozen grading schema."""
import argparse
import fcntl
import json
import sys
from collections import Counter
from pathlib import Path

from common import read, rows, sha
import score_outputs_v2 as scorer
from judge_runtime import load

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FREEZE = ROOT / "configs/formal_v1/freeze.json"

TASKS = ("depiction", "appearance", "content")


def check_results(directory, manifest):
    plan = rows(manifest)
    expected = {r["id"] for r in plan}
    folder = directory / "observations"

    if not (directory / "status.json").exists():
        return False

    if not (directory / "contract.json").exists():
        raise RuntimeError(f"Missing contract: {directory}")

    contract = read(directory / "contract.json")
    if contract["manifest_sha256"] != sha(manifest):
        raise RuntimeError(f"Manifest changed: {directory}")

    files = list(folder.glob("*.json"))
    if {p.stem for p in files} != expected:
        raise RuntimeError(f"Incomplete case IDs: {directory}")

    counts = Counter()
    for p in files:
        r = read(p)
        if r.get("id") != p.stem:
            raise RuntimeError(f"Wrong ID: {p}")
        counts[r.get("status")] += 1

    print(f"{directory}: {dict(counts)}", flush=True)

    if counts["completed"] != len(plan):
        raise RuntimeError(
            f"Grading not qualified: {directory}, {dict(counts)}"
        )

    return read(directory / "status.json").get("status") == "completed"


def smoke_manifest(source):
    output = RESULTS / "final_eval_local_smoke_v1" / "g_cases.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    cases = rows(source)
    explicit = [r for r in cases if r.get("kind") == "explicit"][:6]
    neutral = [r for r in cases if r.get("kind") == "neutral"][:6]

    if len(explicit) != 6 or len(neutral) != 6:
        raise RuntimeError("Cannot construct balanced G smoke cases")

    selected = explicit + neutral
    content = "".join(
        json.dumps(r, ensure_ascii=False) + "\n"
        for r in selected
    )

    if output.exists():
        if output.read_text() != content:
            raise RuntimeError("Existing smoke manifest differs")
    else:
        output.write_text(content)

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=("qwen", "ministral"),
                        required=True)
    parser.add_argument("--mode", choices=("smoke", "full"),
                        required=True)
    args = parser.parse_args()

    lock_path = RESULTS / f"final_eval_local_{args.judge}.lock"

    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        freeze = read(FREEZE)
        states = rows(freeze["states_manifest"])

        if len(states) != 13 or len({s["id"] for s in states}) != 13:
            raise RuntimeError("Expected 13 distinct model states")

        if args.mode == "smoke":
            states = [next(s for s in states if s["id"] == "base")]
            manifest = smoke_manifest(Path(freeze["g_cases"]))
        else:
            manifest = Path(freeze["g_cases"])

        for state in states:
            root = Path(state["aggregate"]).parent
            ready = root / "native_ready.json"

            if not ready.exists():
                raise RuntimeError(f"State not ready: {state['id']}")
            if read(ready)["freeze_sha256"] != sha(FREEZE):
                raise RuntimeError(f"Freeze mismatch: {state['id']}")
            if not Path(state["generation"]).is_dir():
                raise RuntimeError(f"Missing G outputs: {state['id']}")

        print("Loading local judge:", args.judge, flush=True)
        model, processor = load(args.judge)
        print("Loaded:", model.config._name_or_path, flush=True)

        # Reuse the loaded model across states and tasks.
        scorer.load = lambda judge: (model, processor)

        for state in states:
            root = Path(state["aggregate"]).parent

            for task in TASKS:
                if args.mode == "smoke":
                    output = (
                        RESULTS / "final_eval_local_smoke_v1"
                        / args.judge / task
                    )
                else:
                    output = (
                        root / "grades_local_final_v1"
                        / args.judge / task
                    )

                print(
                    "START", state["id"], args.judge, task,
                    flush=True
                )

                if output.exists() and check_results(output, manifest):
                    print("SKIP verified", flush=True)
                    continue

                if output.exists() and not (
                    output / "contract.json"
                ).exists():
                    raise RuntimeError(
                        f"Unsafe existing output: {output}"
                    )

                params = argparse.Namespace(
                    judge=args.judge,
                    task=task,
                    manifest=manifest,
                    measured=Path(state["generation"]),
                    output=output,
                    resume=output.exists(),
                )

                scorer.run(params)

                if not check_results(output, manifest):
                    raise RuntimeError(f"Incomplete scoring: {output}")

                print("DONE", state["id"], task, flush=True)

        print("ALL REQUESTED G EVALUATIONS COMPLETED", flush=True)


if __name__ == "__main__":
    main()
