import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
SPEC = importlib.util.spec_from_file_location("extended", Path(__file__).with_name("extended.py"))
extended = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extended)


class ExtendedTests(unittest.TestCase):
    def test_original_parser_is_extracted_exactly(self):
        parse = extended.pipeline_parser()
        self.assertEqual(parse("Intermediate calculation 13"), 13)
        self.assertIsNone(extended.core.strict_answer("Intermediate calculation 13"))
        self.assertEqual(parse("The answer is 7"), 7)

    def test_candidate_decision_loop_changes_final_mask_gradient(self):
        path = extended.core.ROOT / "GDesigner/graph/graph.py"
        indices = extended.core.gradient_probe(path, "indices")
        result = extended.core.gradient_probe(path, "indices_and_early_decision")
        if result["status"] == "UNAVAILABLE":
            self.skipTest(result["error"])
        self.assertGreater(result["decision_mask_gradient_abs_difference"], 0)
        self.assertGreater(result["nonfinal_mask_gradient_abs_difference"], 0)
        self.assertEqual(indices["decision_mask_gradient_abs_difference"], 0)

    def test_timeout_kind_is_not_inferred_from_error_text(self):
        manifest = {"model": "test", "role_prompt": "Solve", "options": {},
                    "tasks": {"dev": {"problem": "3+4", "truth": 7}},
                    "deadline": time.time() + 5000}
        job = {"block": 0, "condition": "s1_serial", "format": "chat", "limit": 1536,
               "task_id": "dev", "repeat": 0}
        with patch.object(extended.core, "http", side_effect=TimeoutError("timed out")):
            result = extended.request_one(manifest, job, "http://test")
            self.assertEqual(result["error_kind"], "request_timeout")
            manifest["deadline"] = time.time() + 5
            result = extended.request_one(manifest, job, "http://test")
            self.assertEqual(result["error_kind"], "budget_limited_timeout")

    def test_complete_mocked_matrix_and_restart_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "splits.json"
            splits.write_text(json.dumps({"ids": {"dev": ["dev"]},
                                          "tasks": {"dev": {"problem": "3+4", "truth": 7}}}))
            args = argparse.Namespace(out=root / "out", splits=splits, pilot=root / "absent",
                                      tasks=1, deadline=time.time() + 60, model="test", base="http://test")

            def fake_http(base, endpoint, payload=None, timeout=30):
                if endpoint == "/api/tags":
                    return {"models": [{"name": "test", "digest": "same"}]}
                if endpoint == "/api/version":
                    return {"version": "mock"}
                if endpoint == "/api/chat":
                    return {"done": True, "done_reason": "stop", "total_duration": 1000000,
                            "message": {"content": "The answer is 7"}}
                return {"done": True, "done_reason": "stop", "total_duration": 1000000,
                        "response": "The answer is 7"}

            with patch.object(extended.core, "http", fake_http), \
                    patch.object(extended.core, "gradient_probe", return_value={"status": "MOCKED"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                extended.initialize(args)
                for args.block in range(3):
                    for args.slots in (1, 12):
                        extended.run(args)
                summary = extended.summarize(args.out, "TEST_COMPLETED")
                self.assertEqual(summary["recorded"], 108)
                self.assertEqual(summary["planned"], 108)
                self.assertEqual(len(summary["cells"]), 36)
                self.assertEqual(len(summary["restart_comparisons"]), 24)
                for cell in summary["cells"]:
                    self.assertEqual(cell["correct"], 3)
                    self.assertFalse(cell["errors"])
                # Re-entering an already completed block appends no duplicates.
                extended.run(args)
                rows, invalid = extended.core.load_rows(args.out / "responses.jsonl")
                self.assertEqual(len(rows), 108)
                self.assertFalse(invalid)
                self.assertTrue((args.out / "REPORT.txt").exists())
                # A timeout must end this server part, not queue more serial work
                # behind an uncancelled generation.
                manifest = json.loads((args.out / "manifest.json").read_text())
                untouched = root / "timeout"
                untouched.mkdir()
                extended.core.write_json(untouched / "manifest.json", manifest)
                args.out, args.block, args.slots = untouched, 0, 1
                with patch.object(extended, "request_one", return_value={
                    "block": 0, "condition": "s1_serial", "format": "chat", "limit": 1536,
                    "task_id": "dev", "repeat": 0,
                    "error": "TimeoutError: timed out", "error_kind": "request_timeout",
                }) as request:
                    extended.run(args)
                    self.assertEqual(request.call_count, 1)
                status = json.loads((untouched / "summary.json").read_text())["status"]
                self.assertEqual(status, "SERVER_PART_ABORTED_TIMEOUT")

    def test_schedule_matches_across_conditions_and_changes_across_blocks(self):
        manifest = {"task_ids": ["a", "b", "c"], "repeats": 3}
        a = list(extended.schedule(manifest, 0, "s1_serial"))
        b = list(extended.schedule(manifest, 0, "s12_parallel"))
        self.assertEqual([(g[0]["task_id"], g[0]["format"], g[0]["limit"]) for g in a],
                         [(g[0]["task_id"], g[0]["format"], g[0]["limit"]) for g in b])
        self.assertNotEqual(a, list(extended.schedule(manifest, 1, "s1_serial")))


if __name__ == "__main__":
    unittest.main()
