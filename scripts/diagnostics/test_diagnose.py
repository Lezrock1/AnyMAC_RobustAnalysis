import importlib.util
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.error


SPEC = importlib.util.spec_from_file_location("diagnose", Path(__file__).with_name("diagnose.py"))
diagnose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnose)


class DiagnoseTests(unittest.TestCase):
    def test_strict_answer_does_not_score_intermediate_numbers(self):
        self.assertIsNone(diagnose.strict_answer("Intermediate result is 13"))
        self.assertIsNone(diagnose.strict_answer("The answer is 23"))
        self.assertIsNone(diagnose.strict_answer("The answer is 7\nMore reasoning"))
        self.assertEqual(diagnose.strict_answer("Reasoning\nThe answer is 7."), 7)

    def test_router_probe_exposes_actual_token_offsets(self):
        audit = diagnose.router_audit(diagnose.ROOT / "GDesigner/graph/graph.py")
        slices = audit["arun_next_agent_prediction"]
        self.assertTrue(any(s["selected_labels"] == ["task", "role_0", "role_1", "role_2", "role_3"]
                            for s in slices))
        self.assertTrue(any(s["selected_labels"] == ["role_4", "message_0", "message_1"]
                            for s in slices))
        self.assertFalse(audit["gradient_loop_probe"]["decision_step_in_loop"])

    def test_prompt_is_read_without_importing_models(self):
        text = diagnose.role_prompt(diagnose.ROOT / "GDesigner/prompt/masbench_prompt_set.py")
        self.assertIn("quantitative reasoner", text)
        self.assertIn("modulo 23", text)

    def test_real_gradient_method_excludes_decision_mask(self):
        result = diagnose.gradient_probe(diagnose.ROOT / "GDesigner/graph/graph.py")
        if result["status"] == "UNAVAILABLE":
            self.skipTest(result["error"])
        self.assertEqual(result["status"], "MEASURED_SYNTHETIC")
        self.assertGreater(result["nonfinal_mask_gradient_abs_difference"], 0)
        self.assertEqual(result["decision_mask_gradient_abs_difference"], 0)

    def test_audit_preserves_invalid_lines_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "continue_A"
            stage.mkdir()
            path = stage / "train_rounds.jsonl"
            original = '{"round": 4}\n{"round": 4}\n{"broken"\n'
            path.write_text(original)
            entry = diagnose.data_audit(root)["files"]["continue_A/train_rounds.jsonl"]
            self.assertEqual(entry["duplicate_extra_rows"], 1)
            self.assertEqual(entry["invalid"][0]["line"], 3)
            self.assertEqual(path.read_text(), original)

    def test_offline_matrix_saves_responses_and_explicit_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "splits.json"
            splits.write_text(json.dumps({
                "ids": {"dev": ["dev0"]},
                "tasks": {"dev0": {"problem": "Compute 3 + 4 modulo 23.", "truth": 7}},
            }))
            out = root / "out"
            args = argparse.Namespace(
                out=out, splits=splits, pilot=root / "absent", tasks=1,
                model="test-model", deadline=time.time() + 60, base="http://test",
            )
            failures = []

            def fake_http(base, endpoint, payload=None, timeout=30):
                if endpoint == "/api/tags":
                    return {"models": [{"name": "test-model", "digest": "stable"}]}
                if endpoint == "/api/version":
                    return {"version": "test"}
                if (payload["options"]["num_predict"] == 3072
                        and endpoint == "/api/generate" and not failures):
                    failures.append(True)
                    raise urllib.error.URLError("simulated backend failure")
                if endpoint == "/api/chat":
                    self.assertEqual(payload["messages"][0]["role"], "system")
                    return {"done": True, "done_reason": "stop",
                            "message": {"content": "The answer is 7"}}
                self.assertTrue(payload["prompt"].startswith("system: "))
                return {"done": True, "done_reason": "length",
                        "response": "Intermediate result is 7"}

            with contextlib.redirect_stdout(io.StringIO()), patch.object(diagnose, "http", fake_http):
                diagnose.initialize(args)
                for args.slots in (1, 12):
                    diagnose.run(args)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["status"], "COMPLETED_WITH_ERRORS")
            self.assertEqual(summary["recorded_requests"], 48)
            self.assertEqual(summary["planned_requests"], 48)
            self.assertEqual(sum(c["errors"] for c in summary["cells"]), 1)
            rows, invalid = diagnose.load_rows(out / "responses.jsonl")
            self.assertFalse(invalid)
            self.assertEqual(len(rows), 48)
            self.assertTrue(any(r.get("error") for r in rows))
            self.assertTrue((out / "REPORT.txt").exists())


if __name__ == "__main__":
    unittest.main()
