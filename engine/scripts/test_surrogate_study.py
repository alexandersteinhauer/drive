# /// script
# dependencies = ["numpy", "scipy", "soundfile"]
# ///
"""uv run engine/scripts/test_surrogate_study.py"""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_spec = importlib.util.spec_from_file_location("surrogate_study", Path(__file__).with_name("surrogate_study.py"))
s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s)
f = s.fitter


def dense_data(x, weights, taps, zero_dc, preemphasis, mask):
    n, nf = x.shape
    ng = weights.shape[1]
    size = ng*nf*taps
    design = np.column_stack([weights[:, g]*np.r_[np.zeros(k), x[:n-k, feature]]
                              for g in range(ng) for feature in range(nf) for k in range(taps)]
                             + [weights[:, g] for g in range(ng)])
    if zero_dc:
        projection = np.eye(size+ng)
        for offset in range(0, size, taps):
            projection[offset:offset+taps, offset:offset+taps] -= 1/taps
        projection[size:, size:] = 0
        design = design@projection
    return f.lfilter([1, -preemphasis], [1], design, axis=0)*mask[:, None]


class IndependentOperatorTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(764)
        self.taps, self.nf, self.ng = 5, 2, 3
        self.xs = [self.rng.normal(size=(n, self.nf))*[0.1, 4] for n in (19, 31, 43)]
        self.weights = [self.rng.uniform(0.1, 1, size=(len(x), self.ng))*[0.3, 2, 0.8] for x in self.xs]
        self.masks = [np.r_[np.zeros(self.taps), self.rng.uniform(0.1, 3, size=len(x)-self.taps)] for x in self.xs]
        self.dimension = self.ng*self.nf*self.taps+self.ng
        self.diagonal = self.rng.uniform(0.1, 2, size=self.dimension)
        self.baseline = len(self.xs[0])-self.taps

    def test_stacked_dense_objective_and_adjoint_both_backends_and_dc_modes(self):
        u = self.rng.normal(size=self.dimension)
        v = self.rng.normal(size=sum(map(len, self.xs))+self.dimension)
        scale = np.sqrt(self.baseline/sum(np.count_nonzero(mask) for mask in self.masks))
        for zero_dc in (False, True):
            dense = np.concatenate([dense_data(x, w, self.taps, zero_dc, 0.93, mask*scale)
                                    for x, w, mask in zip(self.xs, self.weights, self.masks)]
                                   + [np.diag(self.diagonal)])
            for backend in ("full-fft", "overlap-add"):
                with self.subTest(zero_dc=zero_dc, backend=backend):
                    op = s.IndependentRunsOperator(self.xs, self.weights, self.masks, self.taps, 0.93,
                                                   self.diagonal, self.baseline, backend=backend, block_size=8)
                    op.zero_dc = zero_dc
                    np.testing.assert_allclose(op.matvec(u), dense@u, atol=5e-13, rtol=5e-13)
                    np.testing.assert_allclose(op.rmatvec(v), dense.T@v, atol=5e-13, rtol=5e-13)
                    self.assertAlmostEqual(np.vdot(op.matvec(u), v), np.vdot(u, op.rmatvec(v)), delta=2e-12)
                    np.testing.assert_allclose(op.matmat(u[:, None])[:, 0], dense@u, atol=5e-13)
                    np.testing.assert_allclose(op.rmatmat(v[:, None])[:, 0], dense.T@v, atol=5e-13)
                    ys = [self.rng.normal(size=(len(x), 2)) for x in self.xs]
                    rhs = np.concatenate([f.lfilter([1, -0.93], [1], y[:, 1])*mask*scale
                                          for y, mask in zip(ys, self.masks)]+[np.zeros(self.dimension)])
                    np.testing.assert_allclose(op.target(ys, 1), rhs, atol=1e-13)
                    self.assertAlmostEqual(np.linalg.norm(op.matvec(u)-rhs)**2, np.linalg.norm(dense@u-rhs)**2, delta=2e-10)

    def test_dense_fit_with_original_unit_ridge_and_preconditioning(self):
        ys = [self.rng.normal(size=(len(x), 2)) for x in self.xs]
        fs, gs, penalty = s.preconditioning(self.xs, self.weights, self.taps)
        damping, emphasis = 0.6, 0.87
        scale = np.sqrt(self.baseline/sum(np.count_nonzero(mask) for mask in self.masks))
        events, checkpoints = [], []
        h, b, logs = s.fit_runs([x/fs for x in self.xs], [w/gs for w in self.weights], ys, self.masks,
                                self.taps, damping, emphasis, penalty, self.baseline, zero_dc=[True, False],
                                iterations=500, atol=1e-12, btol=1e-12, progress_every=2,
                                emit=events.append, checkpoint=lambda values, log: checkpoints.append(values.copy()))
        h /= gs[:, None, None, None]*fs[None, None, :, None]
        b /= gs[:, None]
        for output, zero_dc in enumerate((True, False)):
            dense = np.concatenate([dense_data(x, w, self.taps, zero_dc, emphasis, mask*scale)
                                    for x, w, mask in zip(self.xs, self.weights, self.masks)]
                                   + [damping*np.eye(self.dimension)])
            rhs = np.concatenate([f.lfilter([1, -emphasis], [1], y[:, output])*mask*scale
                                  for y, mask in zip(ys, self.masks)] + [np.zeros(self.dimension)])
            expected = np.linalg.lstsq(dense, rhs, rcond=None)[0]
            actual = np.r_[h[:, output].ravel(), b[:, output]]
            np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=3e-9)
            self.assertAlmostEqual(logs[output]["objective"], np.linalg.norm(dense@expected-rhs)**2, delta=2e-10)
            self.assertAlmostEqual(logs[output]["originalCoefficientRidgeSse"], damping**2*np.dot(actual, actual), delta=2e-10)
            self.assertTrue(logs[output]["converged"])
            self.assertLess(logs[output]["relativeNormalResidual"], 1e-9)
        np.testing.assert_allclose(h[:, 0].sum(axis=-1), 0, atol=1e-14)
        np.testing.assert_array_equal(b[:, 0], 0)
        self.assertEqual([len(value) for value in checkpoints], [1, 2])
        self.assertTrue(any(event["event"] == "lsqr_progress" for event in events))

    def test_duplicate_data_changes_neither_objective_preconditioner_nor_solution(self):
        ys = [self.rng.normal(size=(len(x), 1)) for x in self.xs]
        fs, gs, penalty = s.preconditioning(self.xs, self.weights, self.taps)
        dfs, dgs, dpenalty = s.preconditioning(self.xs*2, self.weights*2, self.taps)
        np.testing.assert_allclose(fs, dfs, atol=1e-15)
        np.testing.assert_allclose(gs, dgs, atol=1e-15)
        np.testing.assert_allclose(penalty, dpenalty, atol=1e-14)
        u = self.rng.normal(size=self.dimension)
        operators, results = [], []
        for copies in (1, 2):
            op = s.IndependentRunsOperator(self.xs*copies, self.weights*copies, self.masks*copies,
                                           self.taps, 0.9, self.diagonal, self.baseline)
            op.zero_dc = True
            rhs = op.target(ys*copies, 0)
            operators.append((op, rhs))
            results.append(s.fit_runs([x/fs for x in self.xs]*copies, [w/gs for w in self.weights]*copies,
                                      ys*copies, self.masks*copies, self.taps, 1, 0.9, penalty,
                                      self.baseline, iterations=500, atol=1e-12, btol=1e-12))
        a, arhs = operators[0]
        b, brhs = operators[1]
        np.testing.assert_allclose(a.rmatvec(a.matvec(u)), b.rmatvec(b.matvec(u)), atol=2e-12, rtol=2e-13)
        np.testing.assert_allclose(a.rmatvec(arhs), b.rmatvec(brhs), atol=2e-12, rtol=2e-13)
        self.assertAlmostEqual(np.linalg.norm(a.matvec(u)-arhs)**2, np.linalg.norm(b.matvec(u)-brhs)**2, delta=2e-10)
        np.testing.assert_allclose(results[0][0], results[1][0], atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(results[0][1], results[1][1], atol=2e-9, rtol=2e-9)

    def test_single_baseline_is_the_old_operator_and_fit(self):
        x, weights, mask = self.xs[0], self.weights[0], self.masks[0]
        y = self.rng.normal(size=(len(x), 2))
        with contextlib.redirect_stdout(io.StringIO()):
            old_h, old_b, old_log = f.fit(x, weights, y, self.taps, 500, 1, 0.9, mask, [True, False])
        new_h, new_b, new_log = s.fit_runs([x], [weights], [y], [mask], self.taps, 1, 0.9,
                                          np.ones(self.dimension), self.baseline, zero_dc=[True, False], iterations=500)
        np.testing.assert_allclose(new_h, old_h, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(new_b, old_b, atol=2e-9, rtol=2e-9)
        for old, new in zip(old_log, new_log):
            self.assertEqual(new["stop"], old["stop"])
            self.assertAlmostEqual(new["objective"], old["residual"]**2, delta=1e-10)

    def test_iteration_cap_is_not_convergence(self):
        ys = [self.rng.normal(size=(len(x), 1)) for x in self.xs]
        _, _, logs = s.fit_runs(self.xs, self.weights, ys, self.masks, self.taps, 1, 0.9,
                                np.ones(self.dimension), self.baseline, iterations=1, atol=1e-14, btol=1e-14)
        self.assertFalse(logs[0]["converged"])
        self.assertTrue(logs[0]["iterationLimitReached"])
        self.assertEqual(logs[0]["stopReason"], "iteration_limit")

    def test_convolution_history_resets_at_run_not_concatenation(self):
        xs = [np.r_[np.zeros(5), 1][:, None], np.zeros((6, 1))]
        weights, masks = [np.ones((6, 1))]*2, [np.ones(6)]*2
        op = s.IndependentRunsOperator(xs, weights, masks, 3, 0, np.zeros(4), 12)
        h = np.array([1, 2, 3, 0])
        np.testing.assert_allclose(op.matvec(h)[6:12], 0, atol=1e-14)
        wrong = f._RegressionOperator(f._CausalConvolution(np.concatenate(xs), 3), np.concatenate(weights),
                                      0, np.ones(12), np.zeros(4))
        np.testing.assert_allclose(wrong.matvec(h)[6:8], [2, 3], atol=1e-14)

    def test_preemphasis_and_target_history_reset_independently(self):
        xs = [np.ones((4, 1)), np.full((3, 1), 2.)]
        op = s.IndependentRunsOperator(xs, [np.ones((len(x), 1)) for x in xs],
                                       [np.ones(len(x)) for x in xs], 1, 0.5, np.zeros(2), 7)
        expected = np.r_[1, 0.5, 0.5, 0.5, 2, 1, 1, 0, 0]
        np.testing.assert_allclose(op.matvec([1, 0]), expected, atol=1e-14)
        np.testing.assert_allclose(op.target(xs, 0), expected, atol=1e-14)
        self.assertNotEqual(expected[4], f.lfilter([1, -0.5], [1], np.concatenate(xs)[:, 0])[4])


class MetricsTests(unittest.TestCase):
    def test_pooling_uses_independent_preemphasis_and_energy_weighting(self):
        rng = np.random.default_rng(901)
        targets = [rng.normal(size=(n, 2))*scale for n, scale in ((131, 1), (277, 4))]
        predictions = [target*ratio+rng.normal(size=target.shape)*0.1 for target, ratio in zip(targets, (0.7, 1.1))]
        stats = [s.metric_statistics(t, p, 20000, 5) for t, p in zip(targets, predictions)]
        pooled = s.pooled_metrics(stats, 20000)
        ts = [t.sum(axis=1)[5:] for t in targets]
        ps = [p.sum(axis=1)[5:] for p in predictions]
        expected = sum(np.sum((t-p)**2) for t, p in zip(ts, ps))/sum(t@t for t in ts)
        self.assertAlmostEqual(pooled["normalizedWaveMse"], expected, delta=1e-14)
        ets = [f.lfilter([1, -0.97], [1], t.sum(axis=1))[5:] for t in targets]
        eps = [f.lfilter([1, -0.97], [1], p.sum(axis=1))[5:] for p in predictions]
        expected_e = sum(np.sum((t-p)**2) for t, p in zip(ets, eps))/sum(t@t for t in ets)
        self.assertAlmostEqual(pooled["preemphasizedNormalizedWaveMse"], expected_e, delta=1e-14)
        independent = [f.metrics(t, p, 20000, 5) for t, p in zip(targets, predictions)]
        self.assertAlmostEqual(s.equal_metrics(independent)["waveScore"], np.mean([m["waveScore"] for m in independent]), delta=1e-14)
        self.assertNotAlmostEqual(pooled["waveScore"], s.equal_metrics(independent)["waveScore"], places=4)
        self.assertEqual(pooled["samples"], 398)
        single = s.pooled_metrics(stats[:1], 20000)
        for field in ("normalizedWaveMse", "preemphasizedNormalizedWaveMse", "waveScore", "correlation"):
            self.assertAlmostEqual(single[field], independent[0][field], delta=1e-13)
        for band in single["bands"]:
            self.assertAlmostEqual(single["bands"][band], independent[0]["bands"][band], delta=1e-13)
            self.assertAlmostEqual(single["bandWaveErrorDb"][band], independent[0]["bandWaveErrorDb"][band], delta=1e-13)

    def test_segment_scoring_does_not_reset_preemphasis(self):
        # Prediction becomes perfect at segment 2, but the immediately preceding
        # waveform error still contributes to that segment's causal emphasis.
        y = np.ones((100, 1))
        p = np.r_[np.zeros(50), np.ones(50)][:, None]
        continuous = f.metrics(y, p, 20000, 50)
        wrongly_reset = f.metrics(y[50:], p[50:], 20000)
        self.assertGreater(continuous["preemphasizedNormalizedWaveMse"], 0)
        self.assertEqual(wrongly_reset["preemphasizedNormalizedWaveMse"], 0)


class ManifestAndCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/"reference.json").write_text("{}")
        self.manifest = {"version": 1, "config": "reference.json", "modelRate": 20000,
                         "runs": [{"id": rid, "split": split, "reference": rid+"-reference", "game": rid+"-game", "seed": seed}
                                  for rid, split, seed in (("base", "train", 13), ("added", "train", 14),
                                                           ("val", "validation", 15), ("not-ready", "validation", 16),
                                                           ("final", "test", 99), ("diagnostic", "diagnostic", 100))],
                         "stages": [{"id": "baseline", "trainRuns": ["base"]},
                                    {"id": "targeted", "trainRuns": ["base", "added"]}]}
        self.path = self.root/"manifest.json"
        self.save_manifest()

    def save_manifest(self):
        self.path.write_text(json.dumps(self.manifest))

    def capture(self, rid):
        run = next(run for run in self.manifest["runs"] if run["id"] == rid)
        count, rate, steps = 128, 10000, 4
        program = [[count/rate, 1000, 1000, 0.5]]
        meta = {"rate": rate, "fluidRate": rate*steps, "fluidSteps": steps, "cylinders": 1,
                "global": ["rpm", "manifold", "wastegate", "throttle", "flap"], "cylinder": ["exhaustFlow"],
                "seed": run["seed"], "program": program, "surrogateAudio": True}
        t = np.arange(count)/rate
        flow = 0.1*np.sin(2*np.pi*437*t)+0.03*np.cos(2*np.pi*811*t)
        raw = np.column_stack([np.full(count, 1000), np.full(count, 1e5), np.zeros(count),
                               np.full(count, 0.5), np.zeros(count), flow]).astype("<f4")
        for kind in ("game", "reference"):
            prefix = self.root/run[kind]
            (Path(str(prefix)+".steps.json")).write_text(json.dumps({**meta, "ends": 1 if kind == "reference" else 0}))
            raw.tofile(str(prefix)+".steps.f32")
            if kind == "reference":
                fluid_t = (np.arange(count*steps)+1)/(rate*steps)
                (2*np.sin(2*np.pi*437*fluid_t)+0.3*np.cos(2*np.pi*811*fluid_t)).astype("<f4").tofile(str(prefix)+".ends.f32")

    def run_cli(self, args, error=False):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if error:
                with self.assertRaises(SystemExit) as caught:
                    s.main(args)
                self.assertEqual(caught.exception.code, 2)
            else:
                s.main(args)

    def test_manifest_rejects_holdout_in_stage_seed_overlap_and_capture_aliases(self):
        for failure in ("stage", "seed", "alias"):
            with self.subTest(failure=failure):
                original = json.loads(json.dumps(self.manifest))
                if failure == "stage":
                    self.manifest["stages"][0]["trainRuns"] = ["final"]
                elif failure == "seed":
                    self.manifest["runs"][4]["seed"] = 13
                else:
                    self.manifest["runs"][4]["reference"] = "./base-reference"
                self.save_manifest()
                with self.assertRaises(ValueError):
                    s.load_manifest(self.path)
                self.manifest = original

    def test_wrong_validation_ids_fail_before_captures_or_optimizer(self):
        with mock.patch.object(s, "load_run") as load, mock.patch.object(s, "fit_runs") as fit:
            self.run_cli(["fit", str(self.path), str(self.root/"bad"), "--validation-runs", "final"], error=True)
        load.assert_not_called()
        fit.assert_not_called()

    def test_full_fit_freeze_holdout_and_frozen_tamper_guard(self):
        for rid in ("base", "added", "val"):
            self.capture(rid)
        gate_source = {"rate": 20000, "cylinders": 1, "powers": [1, 2], "flowScale": 0.12,
                       "centers": [[1, 1, 0, 0.5, 0], [2, 1, 0, 0.5, 0]], "widths": [1]*5,
                       "kernels": np.zeros((2, 1, 2, 3)).tolist(), "biases": [[0], [0]], "labels": ["pressure0"]}
        gates = self.root/"gates.json"
        gates.write_text(json.dumps(gate_source))
        out = self.root/"fit"
        with mock.patch.object(f, "metadata", wraps=f.metadata) as metadata:
            self.run_cli(["fit", str(self.path), str(out), "--stage", "targeted", "--taps", "3",
                          "--gates-from", str(gates), "--powers", "3", "--gates", "17", "--iterations", "200",
                          "--atol", "1e-10", "--btol", "1e-10", "--warmup-samples", "8",
                          "--validation-runs", "val", "--progress-every", "2", "--save-predictions"])
        opened = [str(call.args[0]) for call in metadata.call_args_list]
        self.assertFalse(any(name in prefix for prefix in opened for name in ("final", "not-ready", "diagnostic")))
        model_path = Path(str(out)+".model.json")
        model = json.loads(model_path.read_text())
        report = json.loads(Path(str(out)+".report.json").read_text())
        self.assertEqual(model["centers"], gate_source["centers"])
        self.assertEqual(model["widths"], gate_source["widths"])
        self.assertEqual(model["powers"], gate_source["powers"])
        self.assertEqual(model["flowScale"], gate_source["flowScale"])
        self.assertEqual(model["study"]["trainingRuns"], ["base", "added"])
        self.assertEqual(model["study"]["validationRuns"], ["val"])
        self.assertTrue(model["study"]["fitConverged"])
        self.assertEqual(report["warmupPerRun"], 8)
        self.assertEqual(report["optimizer"]["warmupPerRun"], 3)
        normalization = report["optimizer"]["normalization"]
        self.assertEqual(normalization["baselineSamples"], 251)
        self.assertEqual(normalization["effectiveSamples"], 502)
        self.assertAlmostEqual(normalization["dataResidualScale"], 1/np.sqrt(2))
        self.assertEqual(set(report["aggregate"]), {"train", "validation"})
        self.assertEqual(report["aggregate"]["train"]["equalRun"]["count"], 2)
        self.assertEqual(report["aggregate"]["train"]["equalSegment"]["count"], 2)
        self.assertEqual(report["aggregate"]["validation"]["pooled"]["samples"], 246)
        np.testing.assert_allclose(np.asarray(model["kernels"]).sum(axis=-1), 0, atol=1e-14)
        np.testing.assert_array_equal(model["biases"], 0)
        with np.load(str(out)+".checkpoint.npz") as data:
            self.assertEqual(data["solutions"].shape, (1, 14))
            self.assertEqual(json.loads(str(data["metadata"]))["completedOutputs"], 1)
        events = [json.loads(line) for line in Path(str(out)+".progress.jsonl").read_text().splitlines()]
        self.assertEqual(events[0]["event"], "lsqr_start")
        self.assertEqual(events[-1]["event"], "lsqr_done")
        self.assertTrue(any(event["event"] == "lsqr_progress" for event in events))
        for index in range(3):
            self.assertTrue(Path(str(out)+f".run-{index}.npz").exists())
        # Unfrozen test evaluation is rejected before it can read any teacher.
        with mock.patch.object(f, "metadata") as metadata:
            self.run_cli(["evaluate", str(self.path), str(model_path), str(self.root/"bad"), "--split", "test"], error=True)
        metadata.assert_not_called()
        frozen = self.root/"frozen.json"
        with mock.patch.object(f, "metadata") as metadata:
            self.run_cli(["freeze", str(self.path), str(model_path), str(frozen)])
        metadata.assert_not_called()
        self.capture("final")
        with mock.patch.object(s, "lsqr", side_effect=AssertionError("evaluation optimized")), \
             mock.patch.object(s, "kmeans2", side_effect=AssertionError("evaluation reclustered")):
            self.run_cli(["evaluate", str(self.path), str(frozen), str(self.root/"final"), "--split", "test", "--warmup-samples", "8"])
        final = json.loads((self.root/"final.report.json").read_text())
        self.assertEqual([row["id"] for row in final["runs"]], ["final"])
        self.assertIn("not for selection", final["role"])
        self.assertEqual(final["parameterSha256"], report["parameterSha256"])
        tampered = json.loads(frozen.read_text())
        tampered["kernels"][0][0][0][0] += 1
        frozen.write_text(json.dumps(tampered))
        with mock.patch.object(f, "metadata") as metadata:
            self.run_cli(["evaluate", str(self.path), str(frozen), str(self.root/"tampered"), "--split", "test"], error=True)
        metadata.assert_not_called()

    def test_unconverged_model_cannot_silently_be_frozen(self):
        model = {"rate": 20000, "cylinders": 1, "powers": [1], "flowScale": 0.1,
                 "centers": [[0]*5], "widths": [1]*5, "kernels": [[[[0, 0, 0]]]],
                 "biases": [[0]], "labels": ["pressure0"],
                 "study": {"manifestSha256": s.file_digest(self.path), "configSha256": s.file_digest(self.root/"reference.json"),
                           "trainingRuns": ["base"], "validationRuns": ["val"], "fitConverged": False}}
        model_path = self.root/"unconverged.json"
        model_path.write_text(json.dumps(model))
        frozen = self.root/"frozen.json"
        self.run_cli(["freeze", str(self.path), str(model_path), str(frozen)], error=True)
        self.assertFalse(frozen.exists())
        self.run_cli(["freeze", str(self.path), str(model_path), str(frozen), "--allow-unconverged"])
        self.assertTrue(json.loads(frozen.read_text())["study"]["frozen"]["allowUnconverged"])

    def test_external_original_cli_model_can_score_selected_validation_without_fitting(self):
        self.capture("val")
        model = {"rate": 20000, "cylinders": 1, "powers": [1], "flowScale": 0.1,
                 "centers": [[0]*5], "widths": [1]*5, "kernels": [[[[0, 0, 0]]]],
                 "biases": [[0]], "labels": ["pressure0"]}
        path = self.root/"old-model.json"
        path.write_text(json.dumps(model))
        with mock.patch.object(s, "lsqr", side_effect=AssertionError("evaluation optimized")):
            self.run_cli(["evaluate", str(self.path), str(path), str(self.root/"old"), "--split", "validation",
                          "--run-ids", "val", "--warmup-samples", "8"])
        report = json.loads((self.root/"old.report.json").read_text())
        self.assertEqual(report["aggregate"]["validation"]["equalRun"]["count"], 1)
        self.assertIsNone(report["frozen"])


if __name__ == "__main__":
    unittest.main()
