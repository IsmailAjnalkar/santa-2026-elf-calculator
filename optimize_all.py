"""Iterate every local technique over all puzzles until a full pass saves nothing (or the budget runs out).
usage: python optimize_all.py [--hours 2] [--ids 1-1000] [--maxw 8] [--workers 12]
Passes: polish (greedy elimination / dead code / rerouting), resynth (multi-instruction window rebuild),
exact (optimal search for tiny puzzles). Puzzles with the most slack (instructions minus targets) go first.
A puzzle's program replaces the best only if score.check passes and it is strictly shorter; the whole file is
re-validated with score.score before every checkpoint, and a backup is kept for every new best total.
Cache: programs already tried by a technique are skipped. Report: out/optimize_report.csv (per puzzle)."""
import argparse, csv, datetime, hashlib, json, os, shutil, time
from multiprocessing import Pool
from score import load_puzzles, check, score
import polish, resynth, exact

csv.field_size_limit(1 << 26)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
BEST, CACHE, REPORT, BACKUPS = 'out/best.csv', 'out/optimize_cache.json', 'out/optimize_report.csv', 'out/backups'


def h(prog): return hashlib.sha1(json.dumps(prog).encode()).hexdigest()[:16]


def run_polish(args):
    pid, prog, puz, _ = args
    return pid, polish.polish((pid, prog, puz))[1]


def run_resynth(args):
    pid, prog, puz, maxw = args
    return pid, resynth.improve((pid, prog, puz, maxw))[1]


def run_exact(args):
    pid, prog, (s, t, L), _ = args
    if L > 100: return pid, prog                     # exact search only for tiny puzzles
    path = exact.solve_exact(s, t, L, max_depth=len(prog) - 1, budget=20)
    return pid, exact.to_program(s, path) if path else prog


TECHNIQUES = [('polish', run_polish), ('resynth', run_resynth), ('exact', run_exact)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=float, default=2); ap.add_argument('--ids', default='1-1000')
    ap.add_argument('--maxw', type=int, default=8); ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    lo, hi = map(int, a.ids.split('-'))
    P = load_puzzles('data/puzzles.csv')
    rd = lambda: {r['id']: json.loads(r['program']) for r in csv.DictReader(open(BEST, newline='', encoding='utf-8'))}
    S = rd(); start = {k: len(v) for k, v in S.items()}; t_start = time.time()
    total0 = score(BEST, 'data/puzzles.csv')
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    os.makedirs(BACKUPS, exist_ok=True)
    deadline = time.time() + a.hours * 3600
    ids = [k for k in P if lo <= int(k.split('_')[1]) <= hi]
    print(f'start: total {total0}, {len(ids)} puzzles, budget {a.hours} h', flush=True)
    passno = 0
    with Pool(a.workers) as pool:
        while time.time() < deadline:
            pass_saved = 0
            for name, fn in TECHNIQUES:
                ids.sort(key=lambda k: -(len(S[k]) - len(set(P[k][1]) - set(P[k][0]))))   # most slack first
                todo = [k for k in ids if cache.get(f'{name}:{k}') != h(S[k])]
                saved = 0; t0 = time.time()
                for pid, prog in pool.imap_unordered(fn, [(k, S[k], P[k], a.maxw) for k in todo], chunksize=2):
                    if len(prog) < len(S[pid]) and check(prog, *P[pid]) is None:
                        saved += len(S[pid]) - len(prog); S[pid] = prog
                    cache[f'{name}:{pid}'] = h(S[pid])
                    if time.time() > deadline: break
                if saved:   # checkpoint: write, re-validate everything, keep a backup, roll back on failure
                    tmp = BEST + '.opt'
                    cur = rd()   # merge with any newer improvements written by other tools
                    for k in cur:
                        if len(cur[k]) < len(S[k]) and check(cur[k], *P[k]) is None: S[k] = cur[k]
                    with open(tmp, 'w', newline='', encoding='utf-8') as f:
                        w = csv.writer(f); w.writerow(['id', 'program'])
                        for k in sorted(S): w.writerow([k, json.dumps(S[k], separators=(',', ':'))])
                    try:
                        tot = score(tmp, 'data/puzzles.csv')
                        shutil.copy(BEST, os.path.join(BACKUPS, f'best_before_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv'))
                        os.replace(tmp, BEST)
                    except AssertionError as e:
                        print(f'  validation failed ({e}); not saved', flush=True); S = rd(); continue
                json.dump(cache, open(CACHE, 'w'))
                pass_saved += saved
                print(f'pass {passno} {name}: tried {len(todo)}, saved {saved}, {round(time.time() - t0)}s', flush=True)
                if time.time() > deadline: break
            passno += 1
            if pass_saved == 0:
                print('converged: a full pass saved nothing', flush=True); break
    total1 = score(BEST, 'data/puzzles.csv')
    with open(REPORT, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'before', 'after', 'saved'])
        for k in sorted(S): w.writerow([k, start[k], len(S[k]), start[k] - len(S[k])])
    improved = [k for k in S if len(S[k]) < start[k]]
    print(f'done in {round(time.time() - t_start)}s: total {total0} -> {total1} (saved {total0 - total1}), '
          f'{len(improved)} puzzles improved: {improved[:20]}', flush=True)


if __name__ == '__main__':
    main()
