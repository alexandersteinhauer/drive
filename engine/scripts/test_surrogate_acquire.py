# /// script
# dependencies = []
# ///
"""uv run engine/scripts/test_surrogate_acquire.py"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('acquire', Path(__file__).with_name('surrogate_acquire.py'))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads(Path(__file__).with_name('surrogate-data-plan.json').read_text())

    def test_plan_independent_seeds_and_nested_training_stages(self):
        a.validate_plan(self.plan)
        stages = [set(s['trainRuns']) for s in self.plan['stages']]
        self.assertTrue(all(x < y for x, y in zip(stages, stages[1:])))
        totals = [sum(sum(s[0] for s in r['program']) for r in self.plan['runs'] if r['id'] in ids) for ids in stages]
        self.assertAlmostEqual(totals[0], 25.15)
        self.assertAlmostEqual(totals[-1], 81.2)
        first = next(r for r in self.plan['runs'] if r['id'] == 'target-up')
        repeat = next(r for r in self.plan['runs'] if r['id'] == 'target-repeat')
        self.assertEqual(first['program'], repeat['program'])
        self.assertNotEqual(first['seed'], repeat['seed'])

    def test_rejects_leakage_duplicate_ids_seeds_bad_programs_and_missing_jobs(self):
        mutations = [
            lambda p: p['runs'][1].update(seed=p['runs'][0]['seed']),
            lambda p: p['runs'][1].update(id=p['runs'][0]['id']),
            lambda p: p['stages'][0]['trainRuns'].append('test-reapply'),
            lambda p: p['stages'][0]['trainRuns'].append('baseline'),
            lambda p: p['runs'][1]['program'][0].__setitem__(0, 0),
            lambda p: p['runs'][1]['program'][0].__setitem__(1, float('nan')),
            lambda p: p['runs'][1]['program'][0].__setitem__(3, 1.1),
            lambda p: p['acquireOrder'].pop(),
            lambda p: p.update(nativeRate=20000),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                p = copy.deepcopy(self.plan)
                mutation(p)
                with self.assertRaises(ValueError):
                    a.validate_plan(p)

    def make_capture(self, root, teacher):
        prefix = root / ('teacher' if teacher else 'game')
        run = dict(seed=23, program=[[0.001, 4600, 4600, 1]])
        source = {'simulationFrequency': 10000, 'fluidSteps': 4, 'physics': {'cycleVariation': 0.02}}
        reference = {'ducts': {'pipes': []}}
        steps = 24 if teacher else 4
        configured = {**source, 'fluidSteps': steps}
        if teacher:
            configured['ducts'] = reference['ducts']
        meta = dict(rate=10000, fluidSteps=steps, seed=23, program=run['program'], ducted=teacher,
                    upstreamMode='game', simulationSpec=configured, cylinder=['exhaustFlow'], cylinders=8)
        meta['global'] = ['rpm']
        if teacher:
            meta.update(fluidRate=240000, ends=2)
        Path(str(prefix)+'.steps.json').write_text(json.dumps(meta))
        Path(str(prefix)+'.steps.f32').write_bytes(b'\0' * (10 * 9 * 4))
        if teacher:
            for suffix, channels in [('.ends.f32', 2), ('.f32', 1), ('.exits.f32', 6)]:
                Path(str(prefix)+suffix).write_bytes(b'\0' * (10 * 24 * channels * 4))
        return prefix, run, source, reference, meta

    def test_complete_pair_validation_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            for teacher in [False, True]:
                prefix, run, source, reference, _ = self.make_capture(Path(d), teacher)
                result = a.validate_capture(prefix, run, self.plan, source, reference, teacher)
                self.assertEqual(len(result['.steps.f32']['sha256']), 64)
                self.assertEqual(result['.steps.f32']['bytes'], 360)

    def test_rejects_stale_spec_wrong_timing_seed_and_truncated_capture(self):
        with tempfile.TemporaryDirectory() as d:
            for teacher in [False, True]:
                for field, value in [('seed', 24), ('fluidSteps', 12), ('upstreamMode', 'mergedReference'),
                                     ('simulationSpec', {'physics': {'cycleVariation': 0.2}})]:
                    prefix, run, source, reference, meta = self.make_capture(Path(d), teacher)
                    meta[field] = value
                    Path(str(prefix)+'.steps.json').write_text(json.dumps(meta))
                    with self.assertRaises(ValueError):
                        a.validate_capture(prefix, run, self.plan, source, reference, teacher)
                prefix, run, source, reference, _ = self.make_capture(Path(d), teacher)
                Path(str(prefix)+'.steps.f32').write_bytes(b'\0' * 356)
                with self.assertRaises(ValueError):
                    a.validate_capture(prefix, run, self.plan, source, reference, teacher)

    def test_atomic_json_and_content_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'record.json'
            a.write_json(p, {'status': 'complete'})
            self.assertEqual(json.loads(p.read_text()), {'status': 'complete'})
            self.assertFalse(p.with_suffix('.json.tmp').exists())
            self.assertEqual(a.digest(p), a.digest(p))


if __name__ == '__main__':
    unittest.main()
