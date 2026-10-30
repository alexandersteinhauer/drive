# /// script
# dependencies = ["numpy", "scipy", "soundfile"]
# ///
"""uv run engine/scripts/test_surrogate_fit.py"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("surrogate_fit", Path(__file__).with_name("surrogate_fit.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class FitTests(unittest.TestCase):
    def test_power_major_signed_features(self):
        q = np.array([[0.2, -0.3], [0, 0.1]])
        f = s.features(q, [1, 2], 0.1)
        np.testing.assert_allclose(f, [[2, -3, 4, -9], [0, 1, 0, 1]])

    def test_stable_weights(self):
        x = np.array([[0.], [1e6], [-1e6]])
        w = s.gates(x, np.array([[0.], [2.]]), np.array([0.1]))
        self.assertTrue(np.isfinite(w).all())
        np.testing.assert_allclose(w.sum(axis=1), 1)
        self.assertEqual(w[1, 1], 1)
        self.assertEqual(w[2, 0], 1)

    def test_identification_and_current_gate(self):
        rng = np.random.default_rng(12)
        n, taps = 4096, 12
        x = rng.normal(size=(n, 2))
        w = s.gates(np.sin(np.arange(n)[:, None] / 57), np.array([[-0.8], [0.8]]), np.array([0.5]))
        h = rng.normal(size=(2, 2, 2, taps)) * 0.1
        b = np.array([[0.2, -0.4], [-0.3, 0.7]])
        y = s.predict(x, w, h, b)
        mask = np.ones(n); mask[:taps] = 0
        fit, bias, _ = s.fit(x, w, y, taps, 160, 1e-8, 0.9, mask)
        np.testing.assert_allclose(fit, h, atol=2e-6)
        np.testing.assert_allclose(bias, b, atol=2e-6)
        # Independent new inputs/control trajectory, not a neighbouring training window.
        xx = rng.normal(size=x.shape)
        ww = s.gates(np.cos(np.arange(n)[:, None] / 31), np.array([[-0.8], [0.8]]), np.array([0.5]))
        np.testing.assert_allclose(s.predict(xx, ww, fit, bias), s.predict(xx, ww, h, b), atol=1e-5)

    def test_zero_dc_and_preconditioning_preserve_ridge(self):
        rng = np.random.default_rng(5)
        n, taps = 2048, 8
        x = rng.normal(size=(n, 2)) * [0.1, 4]
        w = np.column_stack([np.linspace(0.1, 0.9, n), np.linspace(0.9, 0.1, n)])
        h = rng.normal(size=(2, 1, 2, taps))
        h -= h.mean(axis=-1, keepdims=True)
        y = s.predict(x, w, h, np.zeros((2, 1))) + rng.normal(size=(n, 1)) * 0.02
        mask = np.ones(n); mask[:taps] = 0
        a, ab, _ = s.fit(x, w, y, taps, 350, 1., 0.8, mask, [True])
        xs, gs = np.array([0.1, 4.]), np.array([0.5, 0.5])
        penalty = np.r_[np.repeat((1/(gs[:, None]*xs)).ravel(), taps), 1/gs]
        b, bb, _ = s.fit(x/xs, w/gs, y, taps, 350, 1., 0.8, mask, [True], penalty)
        b /= gs[:, None, None, None]*xs[None, None, :, None]
        bb /= gs[:, None]
        np.testing.assert_allclose(a.sum(axis=-1), 0, atol=1e-12)
        np.testing.assert_allclose(b.sum(axis=-1), 0, atol=1e-12)
        np.testing.assert_allclose(ab, 0)
        np.testing.assert_allclose(bb, 0)
        np.testing.assert_allclose(s.predict(x, w, a, ab), s.predict(x, w, b, bb), atol=3e-5)

    def test_causal_filter(self):
        x = np.zeros((100, 1)); x[10] = 1
        h = np.zeros((1, 1, 1, 25)); h[0, 0, 0, 15] = 2
        y = s.predict(x, np.ones((100, 1)), h, np.zeros((1, 1)))
        np.testing.assert_allclose(y[:25], 0, atol=1e-12)
        self.assertAlmostEqual(y[25, 0], 2)

    def test_actuator_no_future(self):
        g = {"rpm": np.full(200, 1000.), "manifold": np.full(200, 1e5), "wastegate": np.zeros(200), "throttle": np.r_[np.zeros(100), np.ones(100)]}
        cfg = {"ducts": {"flaps": [{"speed": [200, 400], "throttle": [0.3, 0.6]}]}}
        c = s.controls({"rate": 20000}, g, cfg)
        self.assertTrue((c[:100, -1] == 0).all())
        self.assertTrue(c[100, -1] > 0)
        self.assertTrue(c[-1, -1] < 1)
        np.testing.assert_allclose(c[:120], s.controls({"rate": 20000}, {k:v[:120] for k,v in g.items()}, cfg))


if __name__ == "__main__":
    unittest.main()
