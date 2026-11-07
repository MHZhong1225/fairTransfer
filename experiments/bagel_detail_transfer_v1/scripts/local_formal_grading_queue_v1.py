import argparse
import fcntl
import gc
import json
import os
import time
import traceback
from pathlib import Path

import torch
import score_outputs_v2 as scorer
from judge_runtime import load as original_load
from common import rows, sha

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FREEZE = ROOT / "configs/formal_v1/freeze.json"
STATUS = RESULTS / "local_formal_grading_queue_v1.json"


def read(path):
    return json.loads(Path(path).read_text())


def save(**values):
    data = read(STATUS) if STATUS.exists() else {}
    data.update(values)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, STATUS)


def check_models():
    for judge in ("qwen", "ministral"):
        print("Loading", judge, flush=True)
        model, processor = original_load(judge)
        print("OK:", judge, model.device, flush=True)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()


def wait_for_native():
    freeze = read(FREEZE)
    states = rows(freeze["states_manifest"])

    if len(states) != 13:
        raise RuntimeError(f"Expected 13 states, got {len(states)}")

    workers = [
        RESULTS / "formal_worker_v1_gpu0.json",
        RESULTS / "formal_worker_v1_gpu1.json",
    ]

    while True:
        progress = [read(path) for path in workers]
        for item in progress:
            if item["status"] == "failed":
                raise RuntimeError("Formal worker failed; do not grade")

            if item["status"] == "running":
                pid = item.get("pid")
                if pid and not Path(f"/proc/{pid}").exists():
                    raise RuntimeError(
                        f"Worker {pid} disappeared without completion"
                    )

        all_done = all(x["status"] == "completed" for x in progress)
        all_ready = all(
            (Path(s["aggregate"]).parent / "native_ready.json").exists()
            for s in states
        )

        if all_done and all_ready:
            for state in states:
                marker = read(
                    Path(state["aggregate"]).parent / "native_ready.json"
                )
                if marker["freeze_sha256"] != sha(FREEZE):
                    raise RuntimeError("Freeze hash mismatch")
            return freeze, states

        save(
            status="waiting",
            workers=[x["status"] for x in progress],
            ready_states=sum(
                (Path(s["aggregate"]).parent / "native_ready.json").exists()
                for s in states
            ),
            total_states=len(states),
        )
        time.sleep(60)


def grade_all(freeze, states):
    specs = (
        ("understanding", Path(freeze["u_cases"]), "understanding"),
        ("depiction", Path(freeze["g_cases"]), "generation"),
        ("appearance", Path(freeze["g_cases"]), "generation"),
        ("content", Path(freeze["g_cases"]), "generation"),
    )

    original_scorer_load = scorer.load
    finished = []

    for judge in ("qwen", "ministral"):
        model, processor = original_load(judge)
        save(
            status="grading",
            judge=judge,
            model_name=str(
                getattr(model.config, "_name_or_path", "unknown")
            ),
        )

        def cached_load(requested):
            if requested != judge:
                raise ValueError("Judge mismatch")
            return model, processor

        scorer.load = cached_load

        try:
            for state in states:
                for task, manifest, source_key in specs:
                    destination = (
                        Path(state["aggregate"]).parent
                        / "grades_local_v1" / judge / task
                    )

                    if destination.exists() and not (
                        destination / "contract.json"
                    ).exists():
                        raise RuntimeError(
                            f"Incomplete directory without contract: {destination}"
                        )

                    save(
                        status="grading",
                        judge=judge,
                        state=state["id"],
                        task=task,
                    )

                    args = argparse.Namespace(
                        judge=judge,
                        task=task,
                        manifest=manifest,
                        measured=Path(state[source_key]),
                        output=destination,
                        resume=destination.exists(),
                    )
                    scorer.run(args)
                    finished.append(f"{state['id']}/{judge}/{task}")
                    save(finished=finished)
        finally:
            scorer.load = original_scorer_load
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-models", action="store_true")
    args = parser.parse_args()

    if args.check_models:
        check_models()
        return

    lock_path = RESULTS / "local_formal_grading_queue_v1.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        save(status="waiting", pid=os.getpid(), finished=[])
        try:
            freeze, states = wait_for_native()
            grade_all(freeze, states)
            save(status="completed")
        except BaseException:
            save(status="failed", error=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
