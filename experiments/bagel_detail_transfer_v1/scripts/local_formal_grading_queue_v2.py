"""Wait for frozen BAGEL native outputs, then run local independent grading."""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from common import rows, sha

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FREEZE = ROOT / "configs/formal_v1/freeze.json"
STATUS = RESULTS / "local_formal_grading_queue_v2.json"
LOCK = RESULTS / "local_formal_grading_queue_v2.lock"


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def update(**kwargs):
    data = read(STATUS) if STATUS.exists() else {}
    data.update(kwargs, updated_utc=stamp())
    temp = STATUS.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(temp, STATUS)


def native_inputs():
    freeze = read(FREEZE)
    states = rows(Path(freeze["states_manifest"]))
    if len(states) != 13 or len({s["id"] for s in states}) != 13:
        raise RuntimeError("Expected 13 unique states")
    return freeze, states


def wait_for_generation(states):
    worker_files = [
        RESULTS / "formal_worker_v1_gpu0.json",
        RESULTS / "formal_worker_v1_gpu1.json",
    ]

    while True:
        workers = [read(p) for p in worker_files]
        statuses = [w["status"] for w in workers]

        if "failed" in statuses:
            raise RuntimeError("Formal worker failed: " + str(statuses))

        for worker in workers:
            if worker["status"] == "running":
                pid = worker.get("pid")
                if pid and not Path(f"/proc/{pid}").exists():
                    raise RuntimeError(f"Formal worker {pid} disappeared")

        if all(s == "completed" for s in statuses):
            break

        update(status="waiting_for_generation",
               worker_statuses=statuses)
        time.sleep(60)

    for state in states:
        ready = Path(state["aggregate"]).parent / "native_ready.json"
        if not ready.exists():
            raise RuntimeError(f"Missing native_ready: {state['id']}")
        if read(ready)["freeze_sha256"] != sha(FREEZE):
            raise RuntimeError(f"Freeze mismatch: {state['id']}")

        for key in ("understanding", "generation"):
            if not Path(state[key]).exists():
                raise RuntimeError(f"Missing {key}: {state['id']}")

    update(status="native_ready", ready_states=len(states))


def verify_result(directory, manifest):
    if not (directory / "status.json").exists():
        return False
    if not (directory / "contract.json").exists():
        return False

    status = read(directory / "status.json")
    if status.get("status") != "completed":
        return False

    contract = read(directory / "contract.json")
    if contract.get("manifest_sha256") != sha(manifest):
        raise RuntimeError(f"Manifest contract mismatch: {directory}")

    expected = [r["id"] for r in rows(manifest)]
    actual = list((directory / "observations").glob("*.json"))
    if len(actual) != len(expected):
        raise RuntimeError(f"Grading count mismatch: {directory}")

    if {p.stem for p in actual} != set(expected):
        raise RuntimeError(f"Grading IDs mismatch: {directory}")

    missing = 0
    for path in actual:
        item = read(path)
        if item.get("id") != path.stem:
            raise RuntimeError(f"Wrong grading ID: {path}")
        if item.get("status") == "missing_model_output":
            missing += 1

    if missing:
        raise RuntimeError(
            f"{directory}: {missing} missing model outputs"
        )

    return True


def grade(freeze, states):
    tasks = [
        ("understanding", Path(freeze["u_cases"]), "understanding"),
        ("depiction", Path(freeze["g_cases"]), "generation"),
        ("appearance", Path(freeze["g_cases"]), "generation"),
        ("content", Path(freeze["g_cases"]), "generation"),
    ]
    finished = []

    for judge in ("qwen", "ministral"):
        for state in states:
            for task, manifest, measured_key in tasks:
                output = (
                    Path(state["aggregate"]).parent
                    / "grades_local_v1" / judge / task
                )
                key = f"{state['id']}/{judge}/{task}"

                if output.exists() and verify_result(output, manifest):
                    print("SKIP completed:", key, flush=True)
                    finished.append(key)
                    continue

                if output.exists() and not (output / "contract.json").exists():
                    raise RuntimeError(
                        f"Unsafe partial output without contract: {output}"
                    )

                cmd = [
                    sys.executable,
                    str(ROOT / "scripts/score_outputs_v2.py"),
                    "--judge", judge,
                    "--task", task,
                    "--manifest", str(manifest),
                    "--measured", str(Path(state[measured_key])),
                    "--output", str(output),
                ]

                if output.exists():
                    cmd.append("--resume")

                update(
                    status="grading",
                    judge=judge,
                    state=state["id"],
                    task=task,
                    finished_count=len(finished),
                )

                print("START:", key, flush=True)
                subprocess.run(cmd, check=True)

                if not verify_result(output, manifest):
                    raise RuntimeError(f"Incomplete scoring: {key}")

                finished.append(key)
                update(finished_count=len(finished))
                print("DONE:", key, flush=True)

    if len(finished) != 13 * 2 * 4:
        raise RuntimeError("Incomplete task matrix")

    update(status="completed", finished_count=len(finished),
           completed_utc=stamp())


def main():
    with LOCK.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

        update(status="starting", pid=os.getpid(),
               environment=sys.executable,
               started_utc=stamp())

        try:
            freeze, states = native_inputs()
            wait_for_generation(states)
            grade(freeze, states)
        except BaseException:
            update(status="failed", error=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
