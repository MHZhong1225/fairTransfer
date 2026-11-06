"""CPU statistical checks with specified, hand-computable outcomes."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.umm_bias_system import analyze as a
from scripts.umm_bias_system.plan import build_plan, build_codec_plan


def planned_rows():
    return [dict(r, id=r["request_id"], image_id="image_" + r["request_id"], is_repeat=False,
                 repeat_of=None, review_source_status="rendered", generation_status="rendered")
            for r in build_plan("bagel", baseline_scale=4, weak_scale=1)]


def observation(row, judge="qwen", **updates):
    fields = dict(scene_roles=[row["occupation_id"]], person_count=1, presentation="masculine",
                  image_readable=True, severe_artifact=False)
    fields.update(updates)
    return dict(id=row["id"], image_id=row["image_id"], judge=judge, status="annotated",
                model_called=True, json_valid=True, parsed=fields)


class AnalysisTest(unittest.TestCase):
    def test_all_missing_retains_denominator_and_undefined_rd(self):
        plan = [r for r in planned_rows() if r["occupation_id"] == "doctor"
                and r["condition"] == "neutral" and r["guidance"] == "weak"]
        result = a.summarize_cell(a.join_observations(plan, [], "qwen"))
        self.assertEqual(result["N"], 8)
        self.assertEqual(result["J"], 0)
        self.assertIsNone(result["RD"])
        self.assertEqual(result["n_missing_person_count"], 8)
        self.assertEqual(result["n_null_person_count"], 0)

    def test_joint_is_full_plan_and_artifacts_are_separate(self):
        base = next(r for r in planned_rows() if r["condition"] == "perceived_male")
        plan = [dict(base, id=str(i), image_id=str(i)) for i in range(6)]
        raw = [observation(plan[0]), observation(plan[1], person_count=2),
               observation(plan[2], scene_roles=[]), observation(plan[3], image_readable=False),
               observation(plan[4], severe_artifact=None)]
        summary = a.summarize_cell(a.join_observations(plan, raw, "qwen"))
        self.assertEqual(summary["N"], 6)
        self.assertEqual(summary["n_following"], 4)
        self.assertEqual(summary["n_joint"], 2)
        self.assertEqual(summary["n_artifactfree_joint"], 1)
        self.assertEqual(summary["joint_rate"], 2 / 6)
        self.assertEqual(summary["n_null_severe_artifact"], 1)
        self.assertEqual(summary["n_missing_severe_artifact"], 1)

    def test_role_and_artifact_sensitivity_do_not_change_primary(self):
        base = next(r for r in planned_rows() if r["condition"] == "neutral")
        plan = [dict(base, id=str(i), image_id=str(i)) for i in range(4)]
        raw = [observation(plan[0], severe_artifact=True), observation(plan[1], presentation="feminine"),
               observation(plan[2], scene_roles=[], presentation="feminine")]
        summary = a.summarize_cell(a.join_observations(plan, raw, "qwen"))
        self.assertEqual((summary["n_M"], summary["n_F"]), (1, 1))
        self.assertEqual((summary["J"], summary["RD"]), (.5, 0))
        self.assertEqual(summary["n_F_no_role"], 2)
        self.assertEqual(summary["J_artifactfree"], 0)

    def test_legal_null_is_not_missing_and_unknown_is_valid(self):
        row = next(r for r in planned_rows() if r["condition"] == "neutral")
        unknown = a.join_observations([row], [observation(row, person_count=None, severe_artifact=None)], "qwen")[0]
        result = a.summarize_cell([unknown])
        self.assertEqual(result["n_null_person_count"], 1)
        self.assertEqual(result["n_missing_person_count"], 0)
        self.assertEqual(result["n_M"], 0)
        self.assertEqual(a.paired_changes([(unknown, unknown)], "person_count")["same"], 1)
        self.assertEqual(a.paired_changes([(unknown, unknown)], "presentation")["incomparable"], 1)
        one = a.join_observations([row], [observation(row, presentation="unknown")], "qwen")[0]
        self.assertEqual(a.paired_changes([(one, one)], "presentation")["same"], 1)

    def test_missing_bounds_and_role_sets(self):
        row = planned_rows()[0]
        original = a.join_observations([row], [observation(row)], "qwen")[0]
        changed = a.join_observations([row], [observation(row, presentation="feminine")], "qwen")[0]
        missing = a.join_observations([row], [], "qwen")[0]
        summary = a.paired_changes([(original, original), (original, changed), (original, missing)], "presentation")
        self.assertEqual((summary["same"], summary["changed"], summary["incomparable"]), (1, 1, 1))
        self.assertEqual((summary["change_lower"], summary["change_upper"]), (1/3, 2/3))
        left = dict(original, parsed=dict(original["parsed"], person_count=2, scene_roles=["doctor", "nurse"]))
        right = dict(original, parsed=dict(original["parsed"], person_count=0, scene_roles=["nurse", "doctor"]))
        self.assertEqual(a.paired_changes([(left, right)], "scene_roles")["same"], 1)

    def test_readers_remain_separate(self):
        row = next(r for r in planned_rows() if r["condition"] == "neutral")
        left = a.join_observations([row], [observation(row)], "qwen")
        right = a.join_observations([row], [observation(row, judge="phi", presentation="feminine")], "phi")
        cells = a.cell_table(left + right)
        self.assertEqual(len(cells), 2)
        self.assertEqual({(c["judge"], c["n_M"], c["n_F"]) for c in cells}, {("qwen", 1, 0), ("phi", 0, 1)})
        agreement = next(r for r in a.agreement_tables(left + right) if r["field"] == "presentation" and r["condition"] == "all")
        self.assertEqual(agreement["changed"], 1)

    def test_full_group_bootstrap_preserves_identical_arm_contrast(self):
        plan = planned_rows()
        raw = [observation(r, presentation="feminine" if r["sample_index"] >= 2 else "masculine") for r in plan]
        rows = a.join_observations(plan, raw, "qwen")
        records = a.bootstrap_contrasts(rows, replicates=200)
        for row in records:
            if row["comparison"] == "CFG1_minus_CFG4":
                self.assertEqual(row["estimate"], 0)
                self.assertEqual(row["interval_95"], [0, 0])
        weights = a.sampling_weights(200)
        self.assertTrue(np.all(weights.sum(axis=2) == 4))
        self.assertTrue(np.array_equal(weights, a.sampling_weights(200)))

    def test_bootstrap_macro_not_pooled_and_does_not_discard_undefined(self):
        plan = planned_rows()
        raw = [observation(r, presentation="feminine" if r["occupation_id"] in ("doctor", "nurse", "firefighter") else "masculine") for r in plan]
        rows = a.join_observations(plan, raw, "qwen")
        records = a.bootstrap_contrasts(rows, replicates=40)
        baseline = {r["endpoint"]: r for r in records if r["comparison"] == "CFG4"}
        self.assertEqual(baseline["neutral_J_macro"]["estimate"], 0)
        self.assertEqual(baseline["neutral_RD_macro"]["estimate"], 1)
        missing_doctor = [r for r in raw if "__doctor__" not in r["id"]]
        records = a.bootstrap_contrasts(a.join_observations(plan, missing_doctor, "qwen"), replicates=40)
        rd = [r for r in records if r["endpoint"] == "neutral_RD_macro"]
        self.assertTrue(all(r["estimate"] is None and r["interval_95"] is None and r["undefined_replicates"] == 40 for r in rd))

    def test_repeats_are_independent_calls_but_not_extra_images(self):
        original = planned_rows()[0]
        repeat = dict(original, id="repeat", is_repeat=True, repeat_of=original["id"])
        rows = a.join_observations([original, repeat], [observation(original), observation(repeat, presentation="feminine")], "qwen")
        self.assertEqual(len([r for r in rows if not r["is_repeat"]]), 1)
        table = a.repeat_tables(rows)
        self.assertEqual(next(r for r in table if r["field"] == "presentation")["changed"], 1)

    def test_duplicate_observation_is_rejected_not_selected(self):
        row = planned_rows()[0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            a.join_observations([row], [observation(row), observation(row)], "qwen")

    def test_wrong_image_hash_is_rejected(self):
        row = dict(planned_rows()[0], source_image_sha256="expected")
        raw = dict(observation(row), image_sha256="wrong")
        with self.assertRaisesRegex(ValueError, "Wrong observed image hash"):
            a.join_observations([row], [raw], "qwen")

    def test_full_720_image_pipeline_and_no_fake_completion(self):
        source = build_plan("bagel", baseline_scale=4, weak_scale=1)
        source += build_codec_plan(source)
        source += build_plan("sensenova", baseline_scale=4, weak_scale=1)
        plan = [dict(r, id=r["request_id"], image_id="image_" + r["request_id"], is_repeat=False,
                     repeat_of=None, review_source_status="rendered", generation_status="rendered") for r in source]
        repeat_sources = [r for r in plan if r["model"] == "bagel" and r["variant"] == "original"
                          and r["condition"] == "neutral" and r["guidance"] == "native_baseline"
                          and r["template_id"] == 0 and r["sample_index"] == 0]
        plan += [dict(r, id="repeat_" + r["id"], is_repeat=True, repeat_of=r["id"]) for r in repeat_sources]
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study / "PROTOCOL.md").write_text("synthetic test protocol")
            (study / "review/private").mkdir(parents=True)
            (study / "review/private/key.jsonl").write_text("".join(json.dumps(r) + "\n" for r in plan))
            a.prepare(study)
            summary = json.loads(a.run(study).read_text())
            self.assertEqual(summary["n_planned_images"], 720)
            self.assertFalse(summary["complete_review_records"])
            self.assertFalse(summary["formal_review_complete"])
            self.assertEqual(sum(r["N"] for r in summary["cells"]), 1440)
            self.assertTrue(all(r["N_pairs"] == 6 for r in summary["repeat_stability"]))
            pairs = [r for r in summary["paired_changes"] if r["judge"] == "qwen" and r["condition"] == "all"
                     and r["field"] == "scene_roles"]
            self.assertEqual({(r["model"], r["contrast"], r["N_pairs"]) for r in pairs},
                             {("bagel", "codec", 144), ("bagel", "guidance", 144), ("sensenova", "guidance", 144)})

    def test_contract_refuses_inplace_change(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study / "PROTOCOL.md").write_text("fixed protocol")
            first = a.prepare(study)
            self.assertEqual(first, a.prepare(study))
            contract = json.loads(first.read_text())
            self.assertEqual(contract["bootstrap"]["replicates"], 10000)
            (study / "PROTOCOL.md").write_text("changed protocol")
            with self.assertRaisesRegex(ValueError, "already frozen"):
                a.prepare(study)


if __name__ == "__main__":
    unittest.main()
