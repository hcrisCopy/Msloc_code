"""Regression checks for the reported count mismatch and validation-only updates."""
import contextlib
import copy
import io
import sys
import argparse
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from test_contract import ROOT
from extraction_checks import (LEGACY_EXTRACTOR_HASHES, check_frame_count,
                               require_compatible_cache)
from cli_config import parse_stage_args


class ExtractionCheckTests(unittest.TestCase):
    def test_reported_376_frame_case_is_accepted_with_audit(self):
        with self.assertWarnsRegex(RuntimeWarning, "keeping configured FPS=8"):
            report = check_frame_count("reported.mp4", 46.73, 8., 376, 4)
        self.assertAlmostEqual(report["expected_frames"], 373.84)
        self.assertAlmostEqual(report["difference_frames"], 2.16)
        self.assertAlmostEqual(report["difference_seconds"], 0.27)
        self.assertEqual(report["tolerance_frames"], 4)

    def test_explicit_strict_and_large_mismatches_still_fail(self):
        with self.assertRaisesRegex(ValueError, "allowed=2 frames"):
            check_frame_count("strict.mp4", 46.73, 8., 376, 2)
        with self.assertRaisesRegex(ValueError, "FPS/count mismatch"):
            check_frame_count("wrong_fps.mp4", 46.73, 8., 1400, 4)

    def test_known_legacy_cache_is_not_mutated(self):
        legacy = dict(frame_fps=8., sampler="unchanged", checkpoint_sha256="weights-A",
                      extractor_sha256=sorted(LEGACY_EXTRACTOR_HASHES)[0],frame_count_tolerance=2)
        saved = copy.deepcopy(legacy)
        requested = dict(legacy,extractor_sha256="current",frame_count_tolerance=4)
        require_compatible_cache(legacy,requested)
        self.assertEqual(legacy,saved)
        with self.assertRaisesRegex(ValueError,"frame_fps"):
            require_compatible_cache(legacy,dict(requested,frame_fps=4.))
        with self.assertRaisesRegex(ValueError,"checkpoint_sha256"):
            require_compatible_cache(legacy,dict(requested,checkpoint_sha256="weights-B"))
        with self.assertRaisesRegex(ValueError,"Unknown cached extractor"):
            require_compatible_cache(dict(legacy,extractor_sha256="unknown"),requested)

    def test_stage_config_and_cli_override_keep_fps(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder)/"features.yaml"
            config.write_text("train_json_path: train.json\ntest_json_path: test.json\n"
                              "frame_count_tolerance: 7\nframe_fps: 8\n",encoding="utf-8")
            parser = argparse.ArgumentParser()
            parser.add_argument("--annotations",nargs="+",required=True)
            parser.add_argument("--frame-count-tolerance",type=int,default=4)
            parser.add_argument("--frame-fps",type=float,default=8)
            with patch.object(sys,"argv",["extract_features.py","--config",str(config),
                                          "--frame-count-tolerance","9"]):
                args = parse_stage_args(parser,"extract")
            self.assertEqual(args.frame_count_tolerance,9)
            self.assertEqual(args.frame_fps,8)
            self.assertEqual(args.annotations,["train.json","test.json"])


if __name__ == "__main__":
    unittest.main()
