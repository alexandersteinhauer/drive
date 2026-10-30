# /// script
# dependencies = []
# ///
"""Acquire independent, deployment-matched exhaust trajectories from a recorded plan.

uv run engine/scripts/surrogate_acquire.py OUT --renderer PATH --spec PATH \
    [--baseline PREVIOUS_DATA_DIR] [--prepare-only]

No fitting, asset replacement, reference-physics changes or per-record gain. Existing
captures are reusable only after their complete metadata and byte counts are validated.
The immutable manifest is separate from completion records so fitting can start while
later captures are rendered. Large files belong outside the repository.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return {'sha256': h.hexdigest(), 'bytes': Path(path).stat().st_size}


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def validate_plan(plan):
    if plan.get('version') != 1:
        raise ValueError('unsupported plan version')
    ids, seeds = set(), {}
    for r in plan['runs']:
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', r['id']) or r['id'] in ids:
            raise ValueError('invalid/duplicate run id')
        ids.add(r['id'])
        if r['split'] not in ('train', 'validation', 'test', 'diagnostic'):
            raise ValueError('invalid split')
        if not isinstance(r['seed'], int) or not 0 <= r['seed'] <= 2**32-1:
            raise ValueError('invalid seed')
        if r['seed'] in seeds:
            raise ValueError('independent runs require independent seeds')
        seeds[r['seed']] = r['split']
        if not r['program']:
            raise ValueError('empty program')
        for s in r['program']:
            if len(s) != 4 or not all(math.isfinite(v) for v in s) or s[0] <= 0 or min(s[1:3]) <= 0 or not 0 <= s[3] <= 1:
                raise ValueError('invalid program segment')
    for stage in plan['stages']:
        if not stage['trainRuns'] or len(set(stage['trainRuns'])) != len(stage['trainRuns']):
            raise ValueError('invalid stage membership')
        for rid in stage['trainRuns']:
            if rid not in ids or next(r for r in plan['runs'] if r['id'] == rid)['split'] != 'train':
                raise ValueError('stage includes a nontraining run')
    if set(plan['acquireOrder']) != {r['id'] for r in plan['runs'] if not r.get('reuse')} or len(set(plan['acquireOrder'])) != len(plan['acquireOrder']):
        raise ValueError('acquisition order must contain every new run exactly once')
    if (plan['nativeRate'], plan['sourceFluidSteps'], plan['referenceFluidSteps'], plan['modelRate']) != (10000, 4, 24, 20000):
        raise ValueError('changed rates require an explicit acquisition-protocol review')


def validate_capture(prefix, run, plan, source_spec, reference, teacher):
    """Reject partial/stale captures, including a changed upstream spec or sample grid."""
    prefix = str(prefix)
    meta = json.loads(Path(prefix + '.steps.json').read_text())
    steps = plan['referenceFluidSteps'] if teacher else plan['sourceFluidSteps']
    expected = json.loads(json.dumps(source_spec))
    expected['simulationFrequency'], expected['fluidSteps'] = plan['nativeRate'], steps
    if teacher:
        expected['ducts'] = reference.get('ducts', reference)
    else:
        expected.pop('ducts', None)
    frames = sum(math.floor(s[0] * plan['nativeRate'] + 0.5) for s in run['program'])
    if (meta['rate'] != plan['nativeRate'] or meta['fluidSteps'] != steps or
            meta['seed'] != run['seed'] or meta['program'] != run['program'] or
            meta['ducted'] != teacher or meta.get('upstreamMode') != 'game' or
            meta.get('simulationSpec') != expected):
        raise ValueError(f'{prefix}: capture metadata/spec does not match protocol')
    width = len(meta['global']) + meta['cylinders'] * len(meta['cylinder'])
    files = {'.steps.f32': frames * width * 4}
    if teacher:
        if meta['fluidRate'] != plan['nativeRate'] * steps or meta['ends'] != 2:
            raise ValueError(f'{prefix}: unexpected teacher output grid')
        files.update({'.ends.f32': frames * steps * meta['ends'] * 4,
                      '.f32': frames * steps * 4,
                      '.exits.f32': frames * steps * meta['ends'] * 3 * 4})
    for suffix, expected_bytes in files.items():
        if Path(prefix + suffix).stat().st_size != expected_bytes:
            raise ValueError(f'{prefix + suffix}: incomplete byte count')
    return {suffix: digest(prefix + suffix) for suffix in ['.steps.json', *files]}


def prepare(plan, out, renderer, spec_path, baseline):
    validate_plan(plan)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'data').mkdir(exist_ok=True)
    (out / 'captures').mkdir(exist_ok=True)
    for source, name in [(renderer, 'abtrace'), (spec_path, 'spec.json'),
                         (SCRIPT_DIR / plan['config'], 'reference.json')]:
        dest = out / name
        if dest.exists() and digest(source) != digest(dest):
            raise ValueError(f'{dest}: refusing to change pinned acquisition inputs')
        if not dest.exists():
            shutil.copy2(source, dest)
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = {**plan, 'config': str(out / 'reference.json'), 'spec': str(out / 'spec.json'),
                'renderer': str(out / 'abtrace'), 'sourceRevision': revision,
                'identity': {name: digest(out / name) for name in ('abtrace', 'spec.json', 'reference.json')},
                'referenceSource': digest(ROOT / 'engine/src/duct.rs')}
    manifest['runs'] = []
    for run in plan['runs']:
        base = baseline / run['reuse'] if baseline is not None and run.get('reuse') else out / 'data' / run['id']
        manifest['runs'].append({**run, 'game': str(base) + '-game', 'reference': str(base) + '-ref'})
    path = out / 'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        # A later documentation/tool commit is not a new engine or dataset.
        manifest['sourceRevision'] = old['sourceRevision']
        if old != manifest:
            raise ValueError('refusing to overwrite a different study manifest')
    else:
        write_json(path, manifest)
    return manifest


def notify(text):
    subprocess.run(['rad', 'notify', text], check=True, stdout=subprocess.DEVNULL)


def acquire(manifest, out, threads):
    source_spec = json.loads(Path(manifest['spec']).read_text())
    reference = json.loads(Path(manifest['config']).read_text())
    by_id = {r['id']: r for r in manifest['runs']}
    # Validate reused data first; do not spend time rendering against stale baselines.
    order = [r['id'] for r in manifest['runs'] if r.get('reuse')] + manifest['acquireOrder']
    for rid in order:
        run = by_id[rid]
        record = out / 'captures' / (rid + '.json')
        hashes = {}
        for teacher, key in [(False, 'game'), (True, 'reference')]:
            prefix = Path(run[key])
            if Path(str(prefix) + '.steps.json').exists():
                hashes[key] = validate_capture(prefix, run, manifest, source_spec, reference, teacher)
                print(f'{rid} {key}: validated existing capture', flush=True)
                continue
            if run.get('reuse') and prefix.parent != out / 'data':
                raise ValueError(f'missing reused capture: {prefix}')
            command = [manifest['renderer'], manifest['spec'], manifest['config'], str(prefix),
                       'ducted' if teacher else 'lumped',
                       ','.join(':'.join(format(v, '.12g') for v in s) for s in run['program'])]
            env = {**os.environ, 'EXHAUST_ONLY': '1', 'FREQ': str(manifest['nativeRate']),
                   'FLUID': str(manifest['referenceFluidSteps'] if teacher else manifest['sourceFluidSteps']),
                   'SEED': str(run['seed']), 'DUCT_THREADS': str(threads),
                   'SOURCE_REVISION': manifest['sourceRevision']}
            started, last_notice = time.monotonic(), time.monotonic()
            print(f'{rid} {key}: start, {sum(s[0] for s in run["program"]):.2f} simulated seconds', flush=True)
            with Path(str(prefix) + '.log').open('w') as log:
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if teacher and time.monotonic() - last_notice >= 300:
                        notify(f'Expanded exhaust data: {rid}, {line.split(" rpm ")[0].strip()} simulated seconds; renderer progressing.')
                        last_notice = time.monotonic()
                code = process.wait()
            if code:
                raise RuntimeError(f'{rid} {key}: renderer exited {code}; inspect {prefix}.log')
            hashes[key] = validate_capture(prefix, run, manifest, source_spec, reference, teacher)
            print(f'{rid} {key}: complete in {time.monotonic()-started:.1f}s', flush=True)
        write_json(record, {'id': rid, 'seed': run['seed'], 'split': run['split'], 'files': hashes,
                            'acquisitionIdentity': manifest['identity']})
        if not run.get('reuse'):
            notify(f'Expanded exhaust capture {rid} complete: {sum(s[0] for s in run["program"]):.2f}s, independent seed {run["seed"]}.')
    notify('Expanded exhaust dataset complete. Fresh test captures remain reserved for frozen-model evaluation.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', type=Path)
    ap.add_argument('--renderer', required=True, type=Path)
    ap.add_argument('--spec', required=True, type=Path)
    ap.add_argument('--baseline', type=Path)
    ap.add_argument('--plan', type=Path, default=SCRIPT_DIR / 'surrogate-data-plan.json')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    if args.threads <= 0:
        ap.error('--threads must be positive')
    out = args.out.resolve()
    manifest = prepare(json.loads(args.plan.read_text()), out, args.renderer.resolve(), args.spec.resolve(),
                       args.baseline.resolve() if args.baseline else None)
    print(f'manifest: {out / "manifest.json"}', flush=True)
    if not args.prepare_only:
        acquire(manifest, out, args.threads)


if __name__ == '__main__':
    main()
