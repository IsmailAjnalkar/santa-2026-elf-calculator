"""Kaggle-side CP-SAT recombination (goodpjw2008's public recomb_pass.py / recombine.py, unchanged) on our best.
Each pass: LNS with recipe harvesting (elfopt --dump), then free-order CP-SAT recombination per puzzle.
Focus: puzzles of 60-180 instructions (where the public authors saw almost all gains). Passes repeat until
~11 h. Every program is re-validated against the official rules; writes /kaggle/working/submission.csv. Never submits."""
import csv, glob, json, os, shutil, subprocess, sys, time

T_END = time.time() + 11 * 3600
W = '/kaggle/working'
try:
    import ortools  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ortools'], check=True)


def find(name):
    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True))
    assert hits, f'{name} not found'
    return hits[0]


for d in ('src', 'data', 'work', 'bin'): os.makedirs(f'{W}/{d}', exist_ok=True)
for f in ('elf.py', 'pfmt.py', 'recombine.py', 'recomb_pass.py'): shutil.copy(find(f), f'{W}/src/{f}')
shutil.copy(find('puzzles.csv'), f'{W}/data/puzzles.csv')
shutil.copy(find('best.csv'), f'{W}/work/best.csv')
subprocess.run(['g++', '-std=c++17', '-O3', '-march=native', '-o', f'{W}/bin/elfopt', find('elfopt.cpp')], check=True)

sys.path.insert(0, f'{W}/src')
from elf import load_puzzles, read_submission, replay   # noqa: E402
P = load_puzzles(f'{W}/data/puzzles.csv')
S = read_submission(f'{W}/work/best.csv')
base = sum(replay(P[k], S[k]) for k in P)        # raises if anything is invalid
focus = [k for k in P if 60 <= len(S[k]) <= 180]
open(f'{W}/work/focus_ids.txt', 'w').write('\n'.join(focus))
log = open(f'{W}/log.txt', 'w')
jobs = os.cpu_count() or 4
it = 0
while time.time() < T_END - 3 * 3600:          # a pass takes ~2-3 h on 4 cores
    t0 = time.time(); it += 1
    r = subprocess.run([sys.executable, f'{W}/src/recomb_pass.py', '--only', f'{W}/work/focus_ids.txt', '--time', '20',
                        '--cp', '20', '--jobs', str(jobs), '--bin', f'{W}/bin/elfopt', '--extra', '--hall 1'],
                       capture_output=True, text=True, cwd=W)
    S = read_submission(f'{W}/work/best.csv')
    tot = sum(replay(P[k], S[k]) for k in P)      # full re-validation after every pass
    msg = f'pass {it}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]} | total {tot} (start {base}) | {round(time.time() - t0)}s'
    print(msg, flush=True); log.write(msg + '\n'); log.flush()
shutil.copy(f'{W}/work/best.csv', f'{W}/submission.csv')
for d in glob.glob(f'{W}/work/rc_*'): shutil.rmtree(d, ignore_errors=True)
print('done', sum(len(S[k]) for k in P), 'from', base)
