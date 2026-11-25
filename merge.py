"""Keep the shortest valid program per puzzle.  usage: python merge.py out.csv puzzles.csv in1.csv [in2.csv ...]
Inputs are untrusted: every program is re-checked by score.check; bad rows are skipped."""
import csv, json, sys
from score import load_puzzles, check, score

csv.field_size_limit(1 << 26)
out, P = sys.argv[1], load_puzzles(sys.argv[2])
best, src = {}, {}
# "@list.txt" expands to the paths listed in that file (Windows caps a command line at ~32k chars)
inputs = [p for a in sys.argv[3:] for p in (open(a[1:], encoding='utf-8').read().split('\n') if a.startswith('@') else [a]) if p]
for path in inputs:
    ok = total = bad = 0
    try:
        rows = list(csv.DictReader(open(path, newline='', encoding='utf-8-sig')))
    except Exception as e:
        print(f'skip {path}: {e!r}'); continue
    for r in rows:
        try:
            pid, prog = r['id'], json.loads(r['program'])
            if pid not in P or check(prog, *P[pid]) is not None: bad += 1; continue
        except Exception:
            bad += 1; continue
        ok += 1; total += len(prog)
        if pid not in best or len(prog) < len(best[pid]):
            best[pid], src[pid] = prog, path
    print(f'{path}: {ok} valid ({bad} invalid) total {total}')
missing = set(P) - set(best)
assert not missing, f'{len(missing)} puzzles have no valid program, e.g. {sorted(missing)[:3]}'
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['id', 'program'])
    for pid in sorted(best): w.writerow([pid, json.dumps(best[pid], separators=(',', ':'))])
wins = {}
for p in src.values(): wins[p] = wins.get(p, 0) + 1
print('puzzles won per source:', dict(sorted(wins.items(), key=lambda kv: -kv[1])))
print(score(out, sys.argv[2]))
