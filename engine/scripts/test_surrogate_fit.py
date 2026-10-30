# /// script
# dependencies = ["numpy", "scipy", "soundfile"]
# ///
"""uv run engine/scripts/test_surrogate_fit.py"""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

spec = importlib.util.spec_from_file_location("surrogate_fit", Path(__file__).with_name("surrogate_fit.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class ConvolutionOperatorTests(unittest.TestCase):
    def test_both_convolution_backends_match_direct_and_adjoint(self):
        rng = np.random.default_rng(61)
        # Includes exact and partial last blocks, one sample, and taps > samples.
        for n, taps, block in [(1, 1, 1), (7, 11, 12), (23, 7, 8), (32, 8, 8), (33, 8, 8), (63, 9, 12)]:
            x, h, z = rng.normal(size=(n, 2)), rng.normal(size=(3, 2, taps)), rng.normal(size=(n, 3))
            direct = np.column_stack([sum(np.convolve(x[:, f], h[g, f])[:n] for f in range(2)) for g in range(3)])
            gradient = np.zeros_like(h)
            for g in range(3):
                for f in range(2):
                    for k in range(min(n, taps)):
                        gradient[g, f, k] = np.dot(x[:n-k, f], z[k:, g])
            for backend in ["full-fft", "overlap-add"]:
                with self.subTest(n=n, taps=taps, block=block, backend=backend):
                    convolution = s._CausalConvolution(x, taps, backend, block)
                    predicted, adjoint = convolution.forward(h), convolution.adjoint(z)
                    np.testing.assert_allclose(predicted, direct, atol=2e-13, rtol=2e-13)
                    np.testing.assert_allclose(adjoint, gradient, atol=2e-13, rtol=2e-13)
                    self.assertAlmostEqual(np.vdot(predicted, z), np.vdot(h, adjoint), delta=2e-12)

    def test_augmented_operators_match_dense_with_dc_bias_weighting_and_ridge(self):
        rng = np.random.default_rng(62)
        n, nf, ng, taps = 53, 2, 3, 5
        x = rng.normal(size=(n, nf))*[0.1, 4]
        weights = rng.uniform(0.1, 1, size=(n, ng))*[0.3, 2, 0.8]
        mask = rng.uniform(0.1, 3, size=n); mask[:taps] = 0
        preemphasis = 0.93
        size = ng*nf*taps
        diagonal = 0.3*rng.uniform(0.2, 2, size=size+ng)
        design = np.column_stack([weights[:, g]*np.r_[np.zeros(k), x[:n-k, f]]
                                  for g in range(ng) for f in range(nf) for k in range(taps)]
                                 + [weights[:, g] for g in range(ng)])
        u, v = rng.normal(size=size+ng), rng.normal(size=n+size+ng)
        for zero_dc in [False, True]:
            projection = np.eye(size+ng)
            if zero_dc:
                for k in range(0, size, taps):
                    projection[k:k+taps, k:k+taps] -= 1/taps
                projection[size:, size:] = 0
            data = s.lfilter([1, -preemphasis], [1], design@projection, axis=0)*mask[:, None]
            # The prior is applied to the original optimization coordinates, not
            # silently projected/rescaled with the convolutional data operator.
            explicit = np.concatenate([data, np.diag(diagonal)])
            for backend in ["full-fft", "overlap-add"]:
                with self.subTest(zero_dc=zero_dc, backend=backend):
                    convolution = s._CausalConvolution(x, taps, backend, 8)
                    operator = s._RegressionOperator(convolution, weights, preemphasis, mask, diagonal)
                    operator.zero_dc = zero_dc
                    np.testing.assert_allclose(operator.matvec(u), explicit@u, atol=2e-13, rtol=2e-13)
                    np.testing.assert_allclose(operator.rmatvec(v), explicit.T@v, atol=2e-13, rtol=2e-13)
                    self.assertAlmostEqual(np.vdot(operator.matvec(u), v), np.vdot(u, operator.rmatvec(v)), delta=2e-12)
                    # Default matrix actions must use the same vector operators.
                    np.testing.assert_allclose(operator.matmat(u[:, None])[:, 0], explicit@u, atol=2e-13, rtol=2e-13)
                    np.testing.assert_allclose(operator.rmatmat(v[:, None])[:, 0], explicit.T@v, atol=2e-13, rtol=2e-13)

    def test_full_and_blocked_fits_agree(self):
        rng = np.random.default_rng(63)
        n, nf, ng, taps = 4093, 2, 2, 12
        x = rng.normal(size=(n, nf))*[0.2, 3]
        weights = np.column_stack([np.linspace(0.2, 0.9, n), np.linspace(0.8, 0.1, n)])
        h = rng.normal(size=(ng, 2, nf, taps))*0.1
        h[:, 0] -= h[:, 0].mean(axis=-1, keepdims=True)
        bias = np.array([[0, 0.4], [0, -0.7]])
        y = s.predict(x, weights, h, bias)+rng.normal(size=(n, 2))*0.02
        mask = np.linspace(0.1, 2, n); mask[:taps] = 0
        penalty = np.r_[np.repeat([0.5, 2, 3, 0.7], taps), 0.4, 1.3]
        results = []
        for backend in ["full-fft", "overlap-add"]:
            with contextlib.redirect_stdout(io.StringIO()):
                results.append(s.fit(x, weights, y, taps, 300, 0.4, 0.9, mask, [True, False], penalty,
                                     convolution_backend=backend, convolution_block_size=32))
        a, ab, alog = results[0]
        b, bb, blog = results[1]
        np.testing.assert_allclose(a, b, atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose(ab, bb, atol=2e-7, rtol=2e-7)
        np.testing.assert_allclose(s.predict(x, weights, a, ab), s.predict(x, weights, b, bb), atol=2e-6, rtol=2e-6)
        for full_log, block_log in zip(alog, blog):
            self.assertTrue(full_log["converged"])
            self.assertTrue(block_log["converged"])
            self.assertAlmostEqual(full_log["residual"], block_log["residual"], delta=1e-9)
        self.assertEqual(blog[0]["convolution"], {"backend": "overlap-add", "blockSize": 32, "fftSize": 64})

    def test_invalid_block_size_or_backend_rejected(self):
        for block_size in [0, 3, 4.5]:
            with self.subTest(block_size=block_size), self.assertRaisesRegex(ValueError, "integer >= taps"):
                s._CausalConvolution(np.ones((10, 1)), 4, block_size=block_size)
        with self.assertRaisesRegex(ValueError, "unknown convolution backend"):
            s._CausalConvolution(np.ones((10, 1)), 4, backend="unknown")


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
        mask = np.linspace(0.2, 3, n); mask[:taps] = 0
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


class InputTests(unittest.TestCase):
    @staticmethod
    def native():
        meta = {"rate": 10000, "cylinders": 1, "fluidSteps": 4, "surrogateAudio": True}
        glob = {"rpm": np.array([1000., 2000., 4000., 8000.]), "manifold": np.arange(1., 5.)*1e5,
                "wastegate": np.array([0., 0.5, 0.25, 1.]), "throttle": np.array([0., 1., 0., 0.5]),
                "flap": np.array([0.1, 0.3, 0.2, 0.8])}
        return meta, glob, {"exhaustFlow": np.array([[0.], [2.], [-2.], [4.]])}

    def test_10k_to_20k_linear_before_powers_and_gates(self):
        meta, glob, cyl = self.native()
        flow, control, native_flow, native_control = s.model_inputs(meta, glob, cyl, {}, 20000)
        np.testing.assert_array_equal(flow[:, 0], [0, 1, 2, 0, -2, 1])
        np.testing.assert_array_equal(control[:, 0], [1, 1.5, 2, 3, 4, 6])
        np.testing.assert_array_equal(s.features(flow, [1, 2], 1)[:, 1], [0, 1, 4, 0, -4, 1])
        self.assertNotEqual(s.features(flow, [2], 1)[1, 0], s.linear_resample(s.features(native_flow, [2], 1), 10000, 20000)[1, 0])
        centers, widths = np.array([[0.]*5, [1.]*5]), np.ones(5)
        weights = s.gates(control, centers, widths)
        interpolated_weights = s.linear_resample(s.gates(native_control, centers, widths), 10000, 20000)
        self.assertGreater(abs(weights[1, 0]-interpolated_weights[1, 0]), 1e-3)
        self.assertEqual(flow.dtype, np.float32)
        self.assertEqual(control.dtype, np.float32)

    def test_actual_flap_is_respected_even_when_config_would_open_it(self):
        meta, glob, _ = self.native()
        cfg = {"ducts": {"flaps": [{"speed": [0, 1], "throttle": [0, 1]}]}}
        np.testing.assert_array_equal(s.controls(meta, glob, cfg)[:, 4], glob["flap"])
        with self.assertRaises(KeyError):
            s.controls(meta, {k:v for k,v in glob.items() if k != "flap"}, cfg)

    def test_legacy_flap_uses_explicit_fluid_steps_not_recorded_flap(self):
        meta, glob, _ = self.native()
        del meta["surrogateAudio"]
        glob["flap"][:] = 0.9
        cfg = {"ducts": {"flaps": [{"speed": [0, 1], "throttle": [0, 1]}]}}
        control = s.controls(meta, glob, cfg)
        alpha = 1-(1-1/(10000*4*0.15))**4
        np.testing.assert_allclose(control[:, 4], 1-(1-alpha)**np.arange(1, 5), rtol=1e-12)
        self.assertNotEqual(control[0, 4], 1-(1-1/(10000*12*0.15))**12)

    def test_last_native_frame_is_never_extrapolated(self):
        values = np.arange(9, dtype=np.float32)[:, None]
        for native_rate, model_rate in [(10000, 20000), (20000, 20000), (12000, 20000), (30000, 20000), (8000, 20000)]:
            with self.subTest(native_rate=native_rate):
                got = s.linear_resample(values, native_rate, model_rate)
                count = int(np.ceil((len(values)-1)*model_rate/native_rate))
                self.assertEqual(len(got), count)
                np.testing.assert_allclose(got[:, 0], np.arange(count)*native_rate/model_rate, atol=5e-7)
                self.assertLess(got[-1, 0], values[-1, 0])
        with self.assertRaises(ValueError):
            s.linear_resample(values[:1], 10000, 20000)

    def test_packet_boundary_controls_features_match_stream_f32(self):
        rng = np.random.default_rng(123)
        native = rng.normal(size=(521, 7)).astype(np.float32)
        pending, position, outputs, cursor = np.empty((0, 7), dtype=np.float32), 0., [], 0
        step, block = 0.5, 256
        # Mirror Rust's decode_block, including retaining its next left native frame.
        for packet in [1, 17, 129, 3, 256, 115]:
            pending = np.concatenate([pending, native[cursor:cursor+packet]])
            cursor += packet
            while max(0, int(np.ceil((len(pending)-1-position)/step))) >= block:
                for _ in range(block):
                    left = int(np.floor(position))
                    fraction = np.float32(position-left)
                    outputs.append(pending[left] + fraction*(pending[left+1]-pending[left]))
                    position += step
                behind = min(int(np.floor(position)), len(pending)-1)
                pending = pending[behind:]
                position -= behind
        expected = s.linear_resample(native, 10000, 20000)[:len(outputs)]
        np.testing.assert_array_equal(outputs, expected)
        np.testing.assert_array_equal(s.features(np.asarray(outputs)[:, :2], [1, 2], 0.1), s.features(expected[:, :2], [1, 2], 0.1))
        centers = np.array([[-0.2]*5, [0.7]*5])
        np.testing.assert_array_equal(s.gates(np.asarray(outputs)[:, 2:], centers, np.ones(5)), s.gates(expected[:, 2:], centers, np.ones(5)))
        self.assertEqual(len(outputs), 1024)


class TargetTests(unittest.TestCase):
    def test_exact_step_end_grid_new_and_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory)/"teacher")
            for native_rate, fluid_steps, count, first in [(10000, 24, 6, 23), (20000, 12, 3, 11)]:
                with self.subTest(native_rate=native_rate):
                    meta = {"rate": native_rate, "fluidRate": 240000, "fluidSteps": fluid_steps, "ends": 1}
                    raw = np.arange(4*fluid_steps, dtype=np.float32)[:, None]
                    raw.tofile(prefix+".ends.f32")
                    # Isolate the sampling grid from the low-pass filter's boundary response.
                    with mock.patch.object(s, "resample_poly", side_effect=lambda x, up, down, axis: x[::down]):
                        got, labels = s.targets(prefix, meta, count, source_rate=native_rate, model_rate=20000, native_count=4)
                    indices = first + np.arange(count)*12
                    np.testing.assert_array_equal(got[:, 0], indices)
                    np.testing.assert_allclose((indices+1)/240000, 1/native_rate + np.arange(count)/20000, atol=1e-18)
                    self.assertEqual(labels, ["pressure0"])
                    self.assertEqual(s.target_grid(meta, native_rate, 20000), (first, 12))

    def test_antialias_filter_preserves_physical_impulse_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory)/"teacher")
            for native_rate, steps, first in [(10000, 24, 23), (20000, 12, 11)]:
                meta = {"rate": native_rate, "fluidRate": 240000, "fluidSteps": steps, "ends": 1}
                raw = np.zeros((200*steps, 1), dtype=np.float32)
                raw[first+12*100] = 1
                raw.tofile(prefix+".ends.f32")
                count = int(np.ceil((200-1)*20000/native_rate))
                got, _ = s.targets(prefix, meta, count, source_rate=native_rate, model_rate=20000, native_count=200)
                self.assertEqual(np.argmax(got[:, 0]), 100)
                self.assertAlmostEqual(got[100, 0], 1/12, delta=1e-4)

    def test_target_grid_and_recording_mismatch_rejected(self):
        meta = {"rate": 10000, "fluidRate": 240000, "fluidSteps": 24, "ends": 1}
        with self.assertRaisesRegex(ValueError, "native rates differ"):
            s.target_grid(meta, 20000, 20000)
        with self.assertRaisesRegex(ValueError, "integral"):
            s.target_grid(meta, 10000, 44100)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            s.target_grid({**meta, "fluidSteps": 12}, 10000, 20000)
        with self.assertRaisesRegex(ValueError, "--exits requires"):
            s.targets("nonexistent", meta, 10, True, source_rate=10000, model_rate=20000)
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory)/"teacher")
            np.zeros((48, 1), dtype=np.float32).tofile(prefix+".ends.f32")
            with self.assertRaisesRegex(ValueError, "frame counts disagree"):
                s.targets(prefix, meta, 2, source_rate=10000, model_rate=20000, native_count=3)
            with self.assertRaisesRegex(ValueError, "exceeds measured"):
                s.targets(prefix, meta, 4, source_rate=10000, model_rate=20000, native_count=2)

    def test_exit_native_step_average_then_log(self):
        meta = {"rate": 10000, "fluidRate": 40000, "fluidSteps": 4, "ends": 1}
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory)/"teacher")
            np.zeros((16, 1), dtype=np.float32).tofile(prefix+".ends.f32")
            raw = np.column_stack([np.arange(16), 1+np.arange(16), 300+np.arange(16)]).astype(np.float32)
            raw.tofile(prefix+".exits.f32")
            got, labels = s.targets(prefix, meta, 3, True, model_rate=10000, native_count=4)
            averaged = raw.reshape(4, 4, 3).mean(axis=1)[:3]
            np.testing.assert_allclose(got[:, 1], averaged[:, 0])
            np.testing.assert_allclose(got[:, 2:], np.log(averaged[:, 1:]), rtol=1e-7)
            self.assertEqual(labels, ["pressure0", "velocity0", "logDensity0", "logTemperature0"])

    def test_held_out_programs_can_differ_but_each_teacher_source_pair_must_match(self):
        train = {"rate": 10000, "cylinders": 8, "program": [[1, 1000, 1000, 0]], "seed": 17}
        test = {**train, "program": [[0.5, 3000, 5000, 0.7]], "seed": 239}
        rm = {**train, "fluidRate": 240000, "fluidSteps": 24, "ends": 2}
        sm = {**train, "fluidSteps": 4, "surrogateAudio": True}
        tm = {**test, "fluidRate": 240000, "fluidSteps": 24, "ends": 2}
        vm = {**test, "fluidSteps": 4, "surrogateAudio": True}
        counts = (10000, 10000, 5000, 5000)
        s.validate_runs(rm, sm, tm, vm, counts)
        for changed, message in [({"program": train["program"]}, "programs differ"), ({"seed": 17}, "seeds differ"), ({"rate": 20000}, "rates differ")]:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, message):
                s.validate_runs(rm, sm, tm, {**vm, **changed}, counts)
        with self.assertRaisesRegex(ValueError, "frame counts differ"):
            s.validate_runs(rm, sm, tm, vm, (10000, 10000, 5000, 4999))
        with self.assertRaisesRegex(ValueError, "incompatible training/test"):
            s.validate_runs(rm, sm, {**tm, "ends": 3}, vm, counts)

    def test_segments_use_first_source_step_timestamp(self):
        program = [[0.0002, 1000, 1000, 0], [0.0002, 2000, 2000, 1]]
        spans = list(s.segment_ranges(program, 10000., 20000, 6))
        self.assertEqual([(a, end) for _, a, end in spans], [(0, 3), (3, 6)])
        axis = s.time_axis(6, 10000, 20000)
        self.assertEqual(axis["firstTime"], 0.0001)
        self.assertAlmostEqual(axis["lastTime"], 0.00035)
        self.assertAlmostEqual(axis["durationSeconds"], 0.0003)
        self.assertEqual([(a, end) for _, a, end in s.segment_ranges(program, 10000, 10000, 3)], [(0, 2), (2, 3)])


class ObjectiveAndMetricsTests(unittest.TestCase):
    def test_segment_balance_uses_joint_training_pressure_rms(self):
        target = np.array([[1., -1.], [1., -1.], [4., -4.], [4., -4.], [0., 0.], [0., 0.]])
        original = target.copy()
        program = [[1, 1000, 1000, 0], [1, 2000, 2000, 1], [1, 3000, 3000, 0]]
        weights, report = s.balance_segments(target, program, 2, 2)
        np.testing.assert_allclose(weights, [2.5, 2.5, 0.625, 0.625, 1, 1])
        self.assertAlmostEqual(report["medianPositiveJointRmsPa"], 2.5*np.sqrt(2))
        np.testing.assert_array_equal(target, original)
        weights, report = s.balance_segments(np.zeros_like(target), program, 2, 2)
        np.testing.assert_array_equal(weights, np.ones(6))
        self.assertEqual(report["medianPositiveJointRmsPa"], 0)

    def test_weighted_operator_matches_explicit_augmented_ridge(self):
        rng = np.random.default_rng(41)
        n, taps, ng = 96, 4, 2
        x = rng.normal(size=(n, 1))
        w = rng.uniform(0.1, 1, size=(n, ng)); w /= w.sum(axis=1, keepdims=True)
        y = rng.normal(size=(n, 1))
        mask = rng.uniform(0.1, 3, size=n); mask[:taps] = 0
        damping, preemphasis = 0.4, 0.7
        penalty = np.linspace(0.2, 2, ng*taps+ng)
        design = np.column_stack([w[:, g]*np.r_[np.zeros(k), x[:n-k, 0]] for g in range(ng) for k in range(taps)] + [w[:, g] for g in range(ng)])
        matrix = s.lfilter([1, -preemphasis], [1], design, axis=0)*mask[:, None]
        rhs = s.lfilter([1, -preemphasis], [1], y[:, 0])*mask
        explicit = np.linalg.lstsq(np.concatenate([matrix, np.diag(damping*penalty)]), np.r_[rhs, np.zeros(len(penalty))], rcond=None)[0]
        h, b, _ = s.fit(x, w, y, taps, 200, damping, preemphasis, mask, penalty_scale=penalty)
        np.testing.assert_allclose(np.r_[h[:, 0].ravel(), b[:, 0]], explicit, atol=2e-6)

    def test_iteration_limit_is_not_claimed_as_convergence(self):
        rng = np.random.default_rng(9)
        x, y = rng.normal(size=(256, 2)), rng.normal(size=(256, 1))
        w = np.ones((256, 1))
        _, _, log = s.fit(x, w, y, 12, 1, 0.1, 0.9, np.ones(256), [True])
        self.assertEqual(log[0]["stop"], 7)
        self.assertEqual(log[0]["stopReason"], "iteration_limit")
        self.assertEqual(log[0]["iterationLimit"], 1)
        self.assertTrue(log[0]["iterationLimitReached"])
        self.assertFalse(log[0]["converged"])

    def test_band_waveform_error_detects_phase_even_at_identical_levels(self):
        target = np.sin(2*np.pi*500*np.arange(20000)/20000)[:, None]
        result = s.metrics(target, -target, 20000, rpm=3000, cylinders=8)
        self.assertAlmostEqual(result["rmsRatio"], 1)
        self.assertAlmostEqual(result["correlation"], -1)
        self.assertAlmostEqual(result["waveErrorDb"], 10*np.log10(4))
        self.assertAlmostEqual(result["bands"]["400-800"], 0)
        self.assertAlmostEqual(result["bandWaveErrorDb"]["400-800"], 10*np.log10(4))
        order = result["harmonics"]["orders"]["10"]
        self.assertEqual(order["frequencyHz"], 500)
        self.assertAlmostEqual(order["levelRatioDb"], 0)
        self.assertAlmostEqual(order["waveErrorDb"], 10*np.log10(4))
        result = s.metrics(target, target*0.5, 20000)
        self.assertAlmostEqual(result["rmsRatio"], 0.5)
        self.assertAlmostEqual(result["bands"]["400-800"], -10*np.log10(4))
        self.assertAlmostEqual(result["bandWaveErrorDb"]["400-800"], -10*np.log10(4))

    def test_score_uses_continuous_preemphasis_before_segment_slice(self):
        target = np.array([[2.], [0.], [3.], [4.], [-2.], [1.]])
        predicted = np.array([[0.], [1.], [1.], [2.], [-1.], [2.]])
        start = 3
        result = s.metrics(target, predicted, 20000, start)
        raw = np.sum((target[start:]-predicted[start:])**2)/np.sum(target[start:]**2)
        et = s.lfilter([1, -0.97], [1], target[:, 0])[start:]
        ep = s.lfilter([1, -0.97], [1], predicted[:, 0])[start:]
        emphasized = np.sum((et-ep)**2)/np.sum(et**2)
        self.assertAlmostEqual(result["normalizedWaveMse"], raw)
        self.assertAlmostEqual(result["preemphasizedNormalizedWaveMse"], emphasized)
        self.assertAlmostEqual(result["waveScore"], (raw+emphasized)/2)
        reset = s.metrics(target[start:], predicted[start:], 20000)
        self.assertNotAlmostEqual(result["preemphasizedNormalizedWaveMse"], reset["preemphasizedNormalizedWaveMse"])

    def test_zero_and_empty_metrics_are_json_serializable(self):
        for count in [0, 1, 100]:
            result = s.metrics(np.zeros((count, 2)), np.zeros((count, 2)), 20000)
            json.dumps(result, allow_nan=False)
            self.assertEqual(result["samples"], count)


class EndToEndTests(unittest.TestCase):
    def test_cli_exports_model_and_native_rate_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            prefixes = [directory/name for name in ["train-ref", "train-game", "test-ref", "test-game"]]
            for index, prefix in enumerate(prefixes):
                teacher, held_out = index % 2 == 0, index >= 2
                program = [[0.0048, 3000, 3000, 0.4], [0.008, 3000, 3000, 0.4]] if held_out else [[0.0064, 1000, 2000, 1], [0.0064, 2000, 2000, 0.5]]
                meta = {"rate": 10000, "fluidSteps": 24 if teacher else 4, "cylinders": 1,
                        "global": ["rpm", "manifold", "wastegate", "throttle", "flap"], "cylinder": ["exhaustFlow"],
                        "program": program, "seed": 239 if held_out else 17, "surrogateAudio": True}
                if teacher:
                    meta.update({"fluidRate": 240000, "ends": 1})
                Path(str(prefix)+".steps.json").write_text(json.dumps(meta))
                t = (np.arange(128)+1)/10000
                raw = np.column_stack([np.full(128, 3000 if held_out else 1500), np.full(128, 1e5),
                                       np.full(128, 0.2), np.full(128, 0.4), np.linspace(0, 0.7, 128),
                                       0.1*np.sin(2*np.pi*300*t)])
                raw.astype("<f4").tofile(str(prefix)+".steps.f32")
                if teacher:
                    pressure = np.sin(2*np.pi*300*(np.arange(128*24)+1)/240000)
                    pressure.astype("<f4").tofile(str(prefix)+".ends.f32")
            config, out = directory/"reference.json", directory/"model"
            config.write_text("{}")
            argv = ["surrogate_fit.py", *map(str, prefixes), str(config), str(out), "--gates", "1", "--taps", "4",
                    "--iterations", "4", "--gate-width", "1.5", "--balance-segments"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                s.main()
            model = json.loads(Path(str(out)+".model.json").read_text())
            report = json.loads(Path(str(out)+".report.json").read_text())
            inputs = json.loads(Path(str(out)+".inputs.json").read_text())
            with np.load(str(out)+".test.npz") as data:
                self.assertEqual(data["rate"], 20000)
                self.assertEqual(data["nativeRate"], 10000)
                self.assertEqual(data["firstTime"], 0.0001)
                self.assertEqual(data["flow"].shape, (254, 1))
                self.assertEqual(data["controls"].shape, (254, 5))
                self.assertEqual(data["nativeFlow"].shape, (128, 1))
                self.assertEqual(data["nativeControls"].shape, (128, 5))
                for key, suffix in [("flow", ".flow.f32"), ("controls", ".controls.f32"), ("nativeFlow", ".native-flow.f32"), ("nativeControls", ".native-controls.f32")]:
                    np.testing.assert_array_equal(np.fromfile(str(out)+suffix, dtype="<f4").reshape(data[key].shape), data[key])
                combined = s.linear_resample(np.concatenate([data["nativeFlow"], data["nativeControls"]], axis=1), 10000, 20000)
                np.testing.assert_array_equal(combined[:, :1], data["flow"])
                np.testing.assert_array_equal(combined[:, 1:], data["controls"])
            self.assertEqual(model["rate"], 20000)
            self.assertEqual(model["nativeRate"], 10000)
            self.assertEqual(len(model["widths"]), 5)
            self.assertEqual(len(model["centers"][0]), 5)
            self.assertEqual(np.asarray(model["kernels"]).shape, (1, 1, 2, 4))
            np.testing.assert_allclose(np.asarray(model["kernels"]).sum(axis=-1), 0, atol=1e-12)
            self.assertEqual(inputs["nativeRate"], 10000)
            self.assertEqual(inputs["rate"], 20000)
            self.assertEqual(inputs["nativeFrames"], 128)
            self.assertEqual(inputs["frames"], 254)
            self.assertEqual(report["alignment"]["testTeacherFirstIndex"], 23)
            self.assertEqual(report["alignment"]["testTeacherStride"], 12)
            self.assertAlmostEqual(report["trainingAxis"]["durationSeconds"], 254/20000)
            self.assertEqual(report["test"]["samples"], 250)
            self.assertTrue(report["heldOut"]["independentProgram"])
            self.assertTrue(report["heldOut"]["independentSeed"])
            self.assertTrue(report["optimizer"]["segmentBalance"]["enabled"])
            self.assertEqual(report["optimizer"]["gateWidth"], 1.5)
            self.assertEqual(report["optimizer"]["convolution"], {"backend": "overlap-add", "blockSize": 4, "fftSize": 8})
            self.assertEqual(report["segments"][0]["startIndex"], 4)
            self.assertEqual(report["segments"][0]["endIndex"], 95)
            self.assertEqual(report["segments"][1]["startIndex"], 95)
            self.assertAlmostEqual(report["segmentScore"]["mean"], np.mean([segment["waveScore"] for segment in report["segments"]]))
            self.assertEqual(report["segmentScore"]["preemphasis"], 0.97)
            self.assertEqual(report["prefixes"]["test_reference"], str(prefixes[2].resolve()))

            # Even explicit fit-option overrides cannot mutate frozen rate/features/gates.
            evaluation = directory/"evaluation"
            argv = ["surrogate_fit.py", *map(str, prefixes), str(config), str(evaluation),
                    "--evaluate-model", str(out)+".model.json", "--model-rate", "10000", "--gates", "8",
                    "--gate-width", "0.7", "--powers", "3", "--balance-segments"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                 mock.patch.object(s, "fit", side_effect=AssertionError("frozen evaluation must not fit")), \
                 mock.patch.object(s, "kmeans2", side_effect=AssertionError("frozen evaluation must not cluster")):
                s.main()
            frozen_model = json.loads(Path(str(evaluation)+".model.json").read_text())
            frozen_report = json.loads(Path(str(evaluation)+".report.json").read_text())
            self.assertEqual(frozen_model, model)
            self.assertEqual(frozen_report["fit"], [])
            self.assertEqual(frozen_report["optimizer"]["method"], "none: frozen model evaluation")
            self.assertEqual(frozen_report["optimizer"]["powers"], [1, 2])
            self.assertEqual(frozen_report["optimizer"]["gates"], 1)
            self.assertEqual(frozen_report["testAxis"]["rate"], 20000)
            self.assertEqual(frozen_report["segmentScore"], report["segmentScore"])
            for suffix in [".flow.f32", ".controls.f32", ".native-flow.f32", ".native-controls.f32", ".prediction.f32"]:
                self.assertEqual(Path(str(out)+suffix).read_bytes(), Path(str(evaluation)+suffix).read_bytes())


if __name__ == "__main__":
    unittest.main()
