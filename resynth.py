"""Window re-synthesis: rebuild w consecutive instructions with fewer, keeping the rest of the program intact.
usage: python resynth.py in.csv out.csv [maxw] [puzzles.csv]
A window must deliver: targets covered nowhere else, and every register value that later instructions read.
Those values are rebuilt greedily (1 or 2 ops each) from the registers at the window start; accepted only if
the whole program still passes score.check and is shorter."""
import csv, json, sys
from multiprocessing import Pool
from score import load_puzzles, check, NREG

csv.field_size_limit(1 << 26)


def ap(op, x, y):
    if op == '+': return x + y
    if op == '-': return x - y if x > y else 0
    if op == '*': return x * y
    return x // y if x % y == 0 else 0


def one(v, have, limit):
    """(op, x, y) with x op y == v, x and y in `have`."""
    for x in have:
        if v > x and v - x in have: return '+', x, v - x
        if v + x <= limit and v + x in have: return '-', v + x, x
        if v % x == 0 and v // x in have: return '*', x, v // x
        if v * x <= limit and v * x in have: return '/', v * x, x
    return None


def one_table(have, limit):
    """Every value one op from `have` (not already in it) -> (op, x, y)."""
    M, hl = {}, list(have)
    for i, x in enumerate(hl):
        for y in hl[i:]:
            for a, b in ((x, y), (y, x)):
                for op in '+-*/':
                    m = ap(op, a, b)
                    if 0 < m <= limit and m not in have and m not in M:
                        M[m] = (op, a, b)
    return M


def two(v, have, limit, M):
    """[(op, x, y, m), (op, x2, y2, v)]: v in two ops via one new intermediate m from table M."""
    cand = set()
    if v % 2 == 0: cand.add(v // 2)
    r = int(v ** 0.5)
    for q in (r - 1, r, r + 1):
        if q > 0 and q * q == v: cand.add(q)
    for z in have:
        cand.update((v - z, z - v, v + z, v * z))
        if v % z == 0: cand.add(v // z)
        if z % v == 0: cand.add(z // v)
    for m in cand:
        if m in M:
            st = one(v, have | {m}, limit)
            if st and m in st[1:]:
                op, a, b = M[m]
                return [(op, a, b, m), st + (v,)]
    return None


def simulate(prog, start):
    reg = [None] * NREG; reg[:len(start)] = start
    states, vals = [], []
    for op, a, b, d in prog:
        states.append(list(reg))
        v = ap(op, reg[a], reg[b]); vals.append(v); reg[d] = v
    states.append(list(reg))
    return states, vals


def reads_after(prog, j, r):
    """Is register r (as it stands before instruction j) read before being overwritten?"""
    for op, a, b, d in prog[j:]:
        if a == r or b == r: return True
        if d == r: return False
    return False


def attempt(prog, i, w, start, targets, limit, states, vals):
    j = i + w
    win = prog[i:j]
    outside = set(start) | set(vals[:i]) | set(vals[j:])
    need_t = [v for v in dict.fromkeys(vals[i:j]) if v in targets and v not in outside]
    written = {d for _, _, _, d in win}
    live_out = {r: states[j][r] for r in written if reads_after(prog, j, r)}
    keep = {r for r in range(NREG) if r not in written and states[i][r] is not None and reads_after(prog, j, r)}
    need = list(dict.fromkeys(need_t + [v for v in live_out.values()]))
    if len(need) >= w or len(set(live_out.values())) < len(live_out): return None
    have = {v for v in states[i] if v is not None}
    plan, budget = [], w - 1
    pending = [v for v in need if v not in have]
    while pending:  # cheapest-first greedy
        best = None
        for v in pending:
            r = one(v, have, limit)
            if r: best = [r + (v,)]; break
        if best is None:
            M = one_table(have, limit)
            for v in pending:
                best = two(v, have, limit, M)
                if best: break
        if best is None: return None
        plan += best
        if len(plan) > budget: return None
        for st in best: have.add(st[3])
        pending = [v for v in pending if v not in have]
    # registers: live-out values must end in their register; never clobber `keep` or a value still needed
    want = {v: r for r, v in live_out.items()}
    reg = list(states[i])
    ops = []
    for k, (op, x, y, v) in enumerate(plan):
        later = {z for st in plan[k + 1:] for z in st[1:3]}
        try:
            ra, rb = reg.index(x), reg.index(y)
        except ValueError:
            return None
        def safe(r):
            cur = reg[r]
            return r not in keep and (cur is None or cur not in later or reg.count(cur) > 1) \
                and (cur not in want or want[cur] != r or cur == v)
        if v in want and safe(want[v]):
            d = want[v]
        else:
            free = [r for r in range(NREG) if r not in want.values() and safe(r)]
            if not free: return None
            d = free[0]
        reg[d] = v
        ops.append([op, ra, rb, d])
    # live-out values already in place before the window (no op needed) must still sit in their register
    if any(reg[r] != v for r, v in live_out.items()): return None
    new = prog[:i] + ops + prog[j:]
    return new if len(new) < len(prog) and check(new, start, targets, limit) is None else None


def improve(args):
    pid, prog, (start, targets, limit), maxw = args
    tset = set(targets)
    changed = True
    while changed:
        changed = False
        for w in range(2, maxw + 1):
            states, vals = simulate(prog, start)
            for i in range(len(prog) - w + 1):
                new = attempt(prog, i, w, start, tset, limit, states, vals)
                if new:
                    prog, changed = new, True
                    break
            if changed: break
    return pid, prog


if __name__ == '__main__':
    maxw = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    P = load_puzzles(sys.argv[4] if len(sys.argv) > 4 else 'data/puzzles.csv')
    rows = [(r['id'], json.loads(r['program'])) for r in csv.DictReader(open(sys.argv[1], newline='', encoding='utf-8'))]
    before = sum(len(p) for _, p in rows)
    with Pool(12) as pool:
        res = dict(pool.imap_unordered(improve, [(pid, p, P[pid], maxw) for pid, p in rows], chunksize=1))
    after = sum(len(p) for p in res.values())
    with open(sys.argv[2], 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['id', 'program'])
        for pid in sorted(res): w.writerow([pid, json.dumps(res[pid], separators=(',', ':'))])
    print(f'{before} -> {after}  (saved {before - after})')
