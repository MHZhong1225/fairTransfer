import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from common import read, rows, sha

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FREEZE = ROOT / "configs/formal_v1/freeze.json"

freeze = read(FREEZE)
states = rows(freeze["states_manifest"])

assert len(states) == 13, "Expected 13 formal states"

lock_path = RESULTS / "local_qwen_ready_queue.lock"

with lock_path.open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    for state in states:
        name = state["id"]
        root = Path(state["aggregate"]).parent
        ready = root / "native_ready.json"

        print(f"\nChecking {name}", flush=True)

        while not ready.exists():
            workers = [
                read(RESULTS / f"formal_worker_v1_gpu{i}.json")
                for i in (0, 1)
            ]

            if all(w["status"] == "completed" for w in workers):
                raise RuntimeError(f"{name}: native_ready missing")

            if any(w["status"] == "failed" for w in workers):
                raise RuntimeError("Formal worker failed")

            time.sleep(60)

        if read(ready)["freeze_sha256"] != sha(FREEZE):
            raise RuntimeError(f"Freeze mismatch: {name}")

        tasks = [
            ("understanding", freeze["u_cases"], state["understanding"]),
            ("depiction", freeze["g_cases"], state["generation"]),
            ("appearance", freeze["g_cases"], state["generation"]),
            ("content", freeze["g_cases"], state["generation"]),
        ]

        for task, manifest, measured in tasks:
            output = root / "grades_local_v1" / "qwen" / task

            if output.exists():
                if not (output / "contract.json").exists():
                    raise RuntimeError(
                        f"Unsafe existing output: {output}"
                    )

                contract = read(output / "contract.json")
                if (
                    contract.get("judge") != "qwen"
                    or contract.get("task") != task
                    or contract.get("manifest_sha256") != sha(manifest)
                ):
                    raise RuntimeError(
                        f"Grading contract mismatch: {output}"
                    )

                status_file = output / "status.json"
                completed = (
                    status_file.exists()
                    and read(status_file).get("status") == "completed"
                )

                expected_ids = {r["id"] for r in rows(manifest)}
                observed_ids = {
                    p.stem for p in (output / "observations").glob("*.json")
                }

                if completed and expected_ids == observed_ids:
                    print("SKIP", name, task, flush=True)
                    continue

            cmd = [
                sys.executable,
                str(ROOT / "scripts/score_outputs_v2.py"),
                "--judge", "qwen",
                "--task", task,
                "--manifest", str(manifest),
                "--measured", str(measured),
                "--output", str(output),
            ]

            if output.exists():
                cmd.append("--resume")

            print("START", name, task, flush=True)
            subprocess.run(cmd, check=True)
            print("DONE", name, task, flush=True)

print("\nALL QWEN GRADING COMPLETED", flush=True)
