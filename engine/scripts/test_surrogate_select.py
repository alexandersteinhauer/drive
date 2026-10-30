# /// script
# dependencies = ["numpy", "scipy"]
# ///
"""uv run engine/scripts/test_surrogate_select.py"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("surrogate_select", Path(__file__).with_name("surrogate_select.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class SelectionTests(unittest.TestCase):
    def fixture(self, path, name, taps, prediction):
        prefix = str(path/name)
        t = np.arange(3998)/20000+.0001
        reference = np.sin(2*np.pi*1000*t)+.3*np.sin(2*np.pi*300*t)
        np.savez(prefix+".test.npz", reference=reference[:, None], prediction=prediction(reference)[:, None], firstTime=.0001)
        Path(prefix+".model.json").write_text(json.dumps({"rate":20000,"labels":["pressure0"],"kernels":[[[[0.]*taps]]]}))
        Path(prefix+".report.json").write_text(json.dumps({"segmentScore":{"mean":.1},"segments":[{"duration":.2,"rpm":[1000,1000],"throttle":0}]}))
        return prefix

    def test_common_warmup_for_different_filter_lengths(self):
        with tempfile.TemporaryDirectory() as folder:
            p = self.fixture(Path(folder), "short", 32, lambda x: x)
            result = s.score(p, {"rate":10000,"program":[[.2,1000,1000,0]]}, 64)
            self.assertEqual(result["global"]["samples"], 3998-64)
            self.assertAlmostEqual(result["score"], 0)

    def test_spectral_selection_rejects_correct_energy_wrong_phase(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            good=self.fixture(root,"gain",32,lambda x:1.1*x)
            wrong=self.fixture(root,"opposite",64,lambda x:-x)
            meta=root/"validation.json";meta.write_text(json.dumps({"rate":10000,"program":[[.2,1000,1000,0]]}))
            out=root/"selection.json"
            with patch.object(sys,"argv",["select",str(meta),str(out),good,wrong,"--criterion","spectrum"]), contextlib.redirect_stdout(io.StringIO()):
                s.main()
            result=json.loads(out.read_text())
            self.assertEqual(result["winner"],good)
            self.assertEqual(result["commonWarmupSamples"],64)
            self.assertFalse(next(r for r in result["candidates"] if r["prefix"]==wrong)["eligible"])

    def test_different_validation_program_cannot_be_ranked(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.fixture(Path(folder),"model",32,lambda x:x)
            with self.assertRaises(ValueError):
                s.score(p,{"rate":10000,"program":[[.2,2000,2000,0]]},64)


if __name__ == "__main__":
    unittest.main()
