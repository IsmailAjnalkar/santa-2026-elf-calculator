"""Shorten valid programs by deleting an instruction and re-routing its consumers.
usage: python polish.py in.csv out.csv [puzzles.csv] [workers]   (workers default 12; polish_pilot.py used 2)
For instruction k: delete it; every later instruction that read k's result is rewritten to make the same value
in one op from whatever the registers hold at that moment. Accept if the whole program still passes score.check."""
import csv, json, sys
from multiprocessing import Pool
from score import load_puzzles, check, NREG

csv.field_size_limit(1 << 26)


def ap(op, x, y):
    if op == '+': return x + y
    if op == '-': return x - y if x > y else 0
    if op == '*': return x * y
    return x // y if x % y == 0 else 0


def run(prog, start):
    """Register contents before each instruction, plus each instruction's value and source writers."""
    reg = [None] * NREG; reg[:len(start)] = start
    writer = [-1] * NREG                       # instruction index that last wrote each register (-1 = start)
    before, vals, srcs = [], [], []
    for op, a, b, d in prog:
        before.append(list(reg))
        srcs.append((writer[a], writer[b]))
        v = ap(op, reg[a], reg[b])
        vals.append(v)
        reg[d] = v; writer[d] = len(vals) - 1
    return before, vals, srcs


def reroute(v, regs, limit):
    """One instruction [op, i, j] making v from register contents, or None."""
    idx = {}
    for r, x in enumerate(regs):
        if x is not None: idx.setdefault(x, r)
    for x, i in idx.items():
        if v > x and v - x in idx: return ['+', i, idx[v - x]]
        if v + x in idx: return ['-', idx[v + x], i]
        if v % x == 0 and v // x in idx: return ['*', i, idx[v // x]]
        if v * x <= limit and v * x in idx: return ['/', idx[v * x], i]
    return None


def regs_before(prog, i, start):
    reg = [None] * NREG; reg[:len(start)] = start
    for op, a, b, d in prog[:i]:
        if reg[a] is None or reg[b] is None: return None
        reg[d] = ap(op, reg[a], reg[b])
    return reg


def try_delete(prog, k, start, targets, limit):
    _, vals, srcs = run(prog, start)
    users = [u for u in range(k + 1, len(prog)) if k in srcs[u]]
    new = prog[:k] + prog[k + 1:]
    for u in users:                            # u is at index u-1 after the deletion
        reg = regs_before(new, u - 1, start)
        r = reroute(vals[u], reg, limit) if reg else None
        if r is None: return None
        new[u - 1] = r + [new[u - 1][3]]
    return new if check(new, start, targets, limit) is None else None


def polish(args):
    pid, prog, (start, targets, limit) = args
    tset = set(targets)
    improved = True
    while improved:
        improved = False
        vals = run(prog, start)[1]
        for k in range(len(prog) - 1, -1, -1):
            if vals[k] in tset: continue       # target producers are almost never deletable
            new = try_delete(prog, k, start, targets, limit)
            if new is not None:
                prog, improved = new, True
    return pid, prog


if __name__ == '__main__':
    P = load_puzzles(sys.argv[3] if len(sys.argv) > 3 else 'data/puzzles.csv')
    rows = [(r['id'], json.loads(r['program'])) for r in csv.DictReader(open(sys.argv[1], newline='', encoding='utf-8'))]
    before = sum(len(p) for _, p in rows)
    with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 12) as pool:
        res = dict(pool.imap_unordered(polish, [(pid, p, P[pid]) for pid, p in rows], chunksize=4))
    after = sum(len(p) for p in res.values())
    with open(sys.argv[2], 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['id', 'program'])
        for pid in sorted(res): w.writerow([pid, json.dumps(res[pid], separators=(',', ':'))])
    print(f'{before} -> {after}  (saved {before - after})')
