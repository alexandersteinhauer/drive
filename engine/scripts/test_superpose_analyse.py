# /// script
# dependencies = ["numpy", "scipy"]
# ///
"""uv run engine/scripts/test_superpose_analyse.py"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("superpose_analyse", Path(__file__).with_name("superpose_analyse.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class AnalysisTests(unittest.TestCase):
    def test_exact_tiny_factor_names_and_common_f64_precision(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix = str(Path(folder)/"probe")
            factors = [1.0, .9999999, .999999]
            branches = ["A@1", "A@0.9999999", "A@0.999999"]
            n = 4096
            t = np.arange(n)/48000
            base = np.column_stack([np.sin(2*np.pi*1000*t), .3*np.cos(2*np.pi*600*t)])
            source = np.column_stack([np.cos(2*np.pi*900*t), np.sin(2*np.pi*700*t)])
            for name, factor in [("base", 1.0), *zip(branches, factors)]:
                value = base+(1-factor)*source
                value.astype("<f8").tofile(f"{prefix}.{name}.ends.f64")
                # Deliberately uninformative f32 files: the result must come from f64.
                np.zeros_like(value, dtype="<f4").tofile(f"{prefix}.{name}.ends.f32")
            Path(prefix+".json").write_text(json.dumps({
                "rpm":6000,"throttle":1,"record":n/48000,"rate":48000,
                "fluidRate":48000,"ends":2,"burns":{"A":[0,0]},
                "factors":factors,"branches":branches,
            }))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                s.analyse(prefix)
            text = output.getvalue()
            self.assertIn("f64, before directivity", text)
            self.assertIn("unchanged A replay exact: True", text)
            self.assertIn("eps 1e-07", text)
            self.assertIn("next/this norm 10 (linear 10)", text)
            Path(f"{prefix}.{branches[-1]}.ends.f64").unlink()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                s.analyse(prefix)
            self.assertIn("f32, before directivity", output.getvalue())


if __name__ == "__main__":
    unittest.main()
