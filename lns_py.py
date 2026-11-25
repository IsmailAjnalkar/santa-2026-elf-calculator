"""Python cone-LNS (port of the public engine's core move; runs where bin/elfopt.exe is blocked).
usage: python lns_py.py in.csv out.csv [--ids 1-1000] [--sample N] [--secs 60] [--workers 12]
For each target T: remove its exclusive cone (values used only on the way to T), rebuild T from every other value
in the program with a meet-in-the-middle search (mitm.Levels, depth 1: pair/affine shapes, cost <= 5), re-schedule
the value DAG into <= 48 registers, and keep the program only if score.check passes and it is strictly shorter."""
import csv, heapq, json, random, sys, time
from multiprocessing import Pool
from score import load_puzzles, check, NREG

csv.field_size_limit(1 << 26)
OPS = {'+': lambda x, y: x + y, '-': lambda x, y: x - y, '*': lambda x, y: x * y, '/': lambda x, y: x // y}


def ap(op, x, y):
    if op == '+': return x + y
    if op == '-': return x - y if x > y else 0
    if op == '*': return x * y
    return x // y if y and x % y == 0 else 0


def to_dag(prog, start):
    """value -> (op, a, b, position) for the first producer of each value (start values excluded)."""
    reg = [None] * NREG; reg[:len(start)] = start
    node, sset = {}, set(start)
    for k, (op, i, j, d) in enumerate(prog):
        v = ap(op, reg[i], reg[j])
        if v not in node and v not in sset: node[v] = (op, reg[i], reg[j], k)
        reg[d] = v
    return node


def ancestors(v, node, memo):
    if v in memo: return memo[v]
    out = set()
    if v in node:
        op, a, b, _ = node[v]
        for u in (a, b):
            out.add(u); out |= ancestors(u, node, memo)
    memo[v] = out
    return out


def one_step(t, have):
    for x in have:
        if t > x and t - x in have: return '+', x, t - x
        if t + x in have: return '-', t + x, x
        if t % x == 0 and t // x in have: return '*', x, t // x
        if t * x in have: return '/', t * x, x
    return None


def emit(nodes, start, targets, limit):
    """Schedule value DAG {v: (op, a, b, prio)} into a register program (<= 48 live values) or None."""
    sset = set(start)
    users = {}
    indeg = {}
    for v, (op, a, b, pr) in nodes.items():
        deps = {u for u in (a, b) if u in nodes}
        indeg[v] = len(deps)
        for u in deps: users.setdefault(u, []).append(v)
    heap = [(nodes[v][3], v) for v in nodes if indeg[v] == 0]
    heapq.heapify(heap); order = []
    while heap:
        _, v = heapq.heappop(heap); order.append(v)
        for w in users.get(v, []):
            indeg[w] -= 1
            if indeg[w] == 0: heapq.heappush(heap, (nodes[w][3], w))
    if len(order) != len(nodes): return None
    last = {}
    for i, v in enumerate(order):
        _, a, b, _ = nodes[v]
        last[a] = i; last[b] = i
    reg_of, occ = {}, [None] * NREG
    for r, s in enumerate(start):
        if s not in reg_of: reg_of[s] = r; occ[r] = s
    for r, s in enumerate(occ):
        if s is not None and s not in last: occ[r] = None   # never used again: register is free
    prog = []
    for i, v in enumerate(order):
        op, a, b, _ = nodes[v]
        if a not in reg_of or b not in reg_of or occ[reg_of[a]] != a or occ[reg_of[b]] != b: return None
        ra, rb = reg_of[a], reg_of[b]
        for u, r in ((a, ra), (b, rb)):
            if last.get(u) == i: occ[r] = None
        free = next((r for r in range(NREG) if occ[r] is None), None)
        if free is None: return None
        prog.append([op, ra, rb, free]); reg_of[v] = free
        occ[free] = v if last.get(v, -1) > i else None
    return prog


def improve_puzzle(args):
    pid, prog, (start, targets, limit), secs, seed = args
    from mitm import Levels
    rng = random.Random(seed); t_end = time.time() + secs
    best = prog; tried = 0
    while time.time() < t_end:
        node = to_dag(best, start)
        need = [t for t in dict.fromkeys(targets) if t not in set(start)]
        memo = {}
        anc = {t: ancestors(t, node, memo) | {t} for t in need}
        mask = {}
        for t in need:
            for v in anc[t]:
                if v in node: mask.setdefault(v, set()).add(t)
        live = set(mask)                          # values some target depends on
        order = need[:]; rng.shuffle(order)
        improved = False
        for t in order:
            if time.time() > t_end: break
            cone = {v for v in live if mask[v] == {t}}
            if len(cone) < 2: continue
            tried += 1
            desc = {v for v in live if t in anc.get(v, ()) or (v in node and t in ancestors(v, node, memo))}
            pool = [v for v in set(start) | (live - cone) if v not in desc]
            lv = Levels({v: 0 for v in pool}, limit, depth=1)
            r = lv.best(t, len(cone) - 1, affine_a_max=0)   # a from pool values only: L1 x L1 scans explode
            if not r: continue
            plan = r[1]
            have = set(pool); steps = {}
            ok = True
            for v in plan:
                if v in have: continue
                st = one_step(v, have)
                if not st: ok = False; break
                steps[v] = st; have.add(v)
            if not ok or t not in have: continue
            pos = node[t][3]
            nodes = {v: node[v] for v in live - cone}
            for n, (v, (op, a, b)) in enumerate(steps.items()):
                nodes[v] = (op, a, b, pos - 1 + (n + 1) / (len(steps) + 1))
            new = emit(nodes, start, targets, limit)
            if new and len(new) < len(best) and check(new, start, targets, limit) is None:
                best = new; improved = True; break
        if not improved: break
    return pid, best, tried


if __name__ == '__main__':
    args = sys.argv[1:]
    opt = lambda k, d: args[args.index(k) + 1] if k in args else d
    P = load_puzzles('data/puzzles.csv')
    rows = {r['id']: json.loads(r['program']) for r in csv.DictReader(open(args[0], newline='', encoding='utf-8'))}
    lo, hi = map(int, opt('--ids', '1-1000').split('-'))
    ids = [k for k in rows if lo <= int(k.split('_')[1]) <= hi]
    ids.sort(key=lambda k: -(len(rows[k]) - len(P[k][1])))        # most slack first
    if '--sample' in args: ids = ids[:int(opt('--sample', '40'))]
    secs, t0 = float(opt('--secs', '60')), time.time()
    before = sum(len(rows[k]) for k in ids)
    with Pool(int(opt('--workers', '12'))) as pool:
        res = list(pool.imap_unordered(improve_puzzle, [(k, rows[k], P[k], secs, i) for i, k in enumerate(ids)]))
    out = dict(rows); improved = []
    for pid, prog, _ in res:
        if len(prog) < len(rows[pid]) and check(prog, *P[pid]) is None:
            out[pid] = prog; improved.append((pid, len(rows[pid]), len(prog)))
    with open(args[1], 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['id', 'program'])
        for pid in sorted(out): w.writerow([pid, json.dumps(out[pid], separators=(',', ':'))])
    after = sum(len(out[k]) for k in ids)
    print(f'{len(ids)} puzzles, {sum(r[2] for r in res)} cones tried, {before} -> {after} (saved {before - after}) '
          f'in {round(time.time() - t0)}s; improved: {improved[:15]}')
