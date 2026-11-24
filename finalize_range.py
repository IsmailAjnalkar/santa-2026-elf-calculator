"""Build the updated CSV for a puzzle range and print the summary.
usage: python finalize_range.py <input.csv> <candidate.csv> <out.csv> <lo-hi>
Takes a puzzle's program from the candidate only if it is in lo-hi, valid, and strictly shorter;
every other puzzle keeps its input program unchanged (verified)."""
import csv, json, sys
from score import load_puzzles, check, score

csv.field_size_limit(1 << 26)
inp, cand, out, rng = sys.argv[1:5]
lo, hi = map(int, rng.split('-'))
P = load_puzzles('data/puzzles.csv')
read = lambda p: {r['id']: r['program'] for r in csv.DictReader(open(p, newline='', encoding='utf-8'))}
A, B = read(inp), read(cand)

final, improved = dict(A), []
for pid in sorted(P):
    if not lo <= int(pid.split('_')[1]) <= hi or pid not in B: continue
    old, new = json.loads(A[pid]), json.loads(B[pid])
    if len(new) < len(old) and check(new, *P[pid]) is None:
        final[pid] = B[pid]; improved.append((pid, len(old), len(new)))

outside = [pid for pid in P if not lo <= int(pid.split('_')[1]) <= hi]
assert all(final[pid] == A[pid] for pid in outside), 'a program outside the range changed'
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['id', 'program'])
    for pid in sorted(P): w.writerow([pid, final[pid]])

rng_total = lambda S: sum(len(json.loads(S[f'scored_{i:04d}'])) for i in range(lo, hi + 1))
print(f'range {lo}-{hi}: old {rng_total(A)}  new {rng_total(final)}  (saved {rng_total(A) - rng_total(final)})')
print(f'whole file: old {score(inp, "data/puzzles.csv")}  new {score(out, "data/puzzles.csv")}  (all 1000 programs valid)')
print(f'{len(improved)} puzzles improved:')
for pid, o, n in improved: print(f'  {pid}: {o} -> {n}')
