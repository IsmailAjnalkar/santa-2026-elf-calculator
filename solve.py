"""Greedy solver.  usage: python solve.py puzzles.csv out.csv [--limit N]
1) emit any target reachable in one op from register values;
2) else emit the intermediate that makes the most targets one-op reachable;
3) else build the smallest target by double-and-add from the nearest known value.
"""
import csv, json, sys
from multiprocessing import Pool
import numpy as np
from score import load_puzzles, check, NREG


def ap(op, x, y):
    if op == '+': return x + y
    if op == '-': return x - y if x > y else None
    if op == '*': return x * y
    return x // y if x % y == 0 else None


class S:
    def __init__(self, start, targets, limit):
        self.reg = [None] * NREG
        self.reg[:len(start)] = start
        self.last = [0] * NREG          # last-touched step, for LRU eviction
        self.prog, self.limit = [], limit
        self.need = set(targets) - set(start)

    def vals(self):
        return {v: i for i, v in enumerate(self.reg) if v is not None}

    def emit(self, op, i, j, d=None, keep=()):
        v = ap(op, self.reg[i], self.reg[j])
        assert v and 0 < v <= self.limit, (op, i, j, v)
        if d is None:
            free = [r for r in range(NREG) if self.reg[r] is None]
            # ponytail: plain LRU eviction; smarter liveness-aware allocation if registers ever get tight
            d = free[0] if free else min((r for r in range(NREG) if r not in keep and r not in (i, j)),
                                         key=self.last.__getitem__)
        t = len(self.prog) + 1
        self.last[i] = self.last[j] = self.last[d] = t
        self.reg[d] = v
        self.prog.append([op, i, j, d])
        self.need.discard(v)
        return d


def one_step(t, V):
    """(op, value_a, value_b) with a op b == t, both in V, or None."""
    for x in V:
        if t - x in V and t > x: return '+', x, t - x
        if t + x in V: return '-', t + x, x
        if t % x == 0 and t // x in V: return '*', x, t // x
        if t * x in V: return '/', t * x, x
    return None


def combine(X, Y, limit):
    """Every x op y (both operand orders) for x∈X, y∈Y as an int64 array; positive, ≤ limit."""
    x, y = X[:, None], Y[None, :]
    s = x + y
    out = [s[s <= limit], np.abs(x - y)[x != y]]
    ok = x <= limit // y
    out.append((x * y)[ok])  # masked entries may overflow; they are discarded
    out.append((x // y)[x % y == 0])
    out.append((y // x)[y % x == 0])
    return np.concatenate(out)


def isin(sorted_arr, q):
    if len(sorted_arr) == 0: return np.zeros(len(q), bool)
    i = np.minimum(np.searchsorted(sorted_arr, q), len(sorted_arr) - 1)
    return sorted_arr[i] == q


class Tiers:
    """Register values (cost 0) and every value 1 and 2 instructions away, as sorted arrays."""
    def __init__(self, V, limit):
        self.limit = limit
        self.T0 = np.array(sorted(V), dtype=np.int64)
        self.T1 = np.setdiff1d(combine(self.T0, self.T0, limit), self.T0)
        T1 = self.T1
        two = [combine(T1, self.T0, limit), (2 * T1)[T1 <= limit // 2], (T1 * T1)[T1 <= int(limit ** 0.5)]]
        self.T2 = np.setdiff1d(np.concatenate(two), np.union1d(self.T0, T1))

    def cost(self, v):
        for c, T in enumerate((self.T0, self.T1, self.T2)):
            if isin(T, np.array([v]))[0]: return c
        return None

    def mitm(self, t, a, B):
        """(a_i, b) with a_i op b == t for some a_i in array a and b in sorted array B, or None."""
        L = self.limit
        rels = [(t - a, a < t), (a - t, a > t), (t + a, a <= L - t),
                (t // a, t % a == 0), (a // t, a % t == 0), (t * a, a <= L // t)]
        for b, m in rels:
            hit = m & isin(B, b)
            if hit.any():
                i = np.flatnonzero(hit)[0]
                return int(a[i]), int(b[i])
        return None

    def pair(self, t, max_cost=4):
        """t = a op b with a,b each 1-2 ops away: T1⊕T1 (3), T1⊕T2 (4), T2⊕T2 (5).  -> (cost, a, b) or None."""
        for c, a, B in ((3, self.T1, self.T1), (4, self.T1, self.T2), (5, self.T2, self.T2)):
            if c > max_cost: break
            r = self.mitm(t, a, B)
            if r: return (c,) + r
        return None

    def affine(self, t):
        """t = A*B ± C with A∈T0, B∈T1, C∈T0/T1/T2 (cost 3/4/5).  -> (cost, A, B, C) or None."""
        A, B, L = self.T0[:, None], self.T1[None, :], self.limit
        ok = A <= L // B
        p = (A * B)
        p = np.where(ok, p, 0)
        for c, T in ((3, self.T0), (4, self.T1), (5, self.T2)):
            for C, m in ((t - p, ok & (p < t)), (p - t, ok & (p > t))):
                hit = m & isin(T, C.ravel()).reshape(C.shape)
                if hit.any():
                    i, j = np.argwhere(hit)[0]
                    return c, int(self.T0[i]), int(self.T1[j]), int(C[i, j])
        return None


def plan_of(v, V, M):
    """Values to emit, in order, so that v exists (each one op from V and earlier ones)."""
    if v in V: return []
    if v in M: return [v]
    m = next(m for m in partners(v, V) if m in M)
    return [m, v]


def emit_plan(st, plan):
    V = st.vals()
    have, steps = dict(V), []
    for v in plan:
        if v in have: continue
        hit = one_step(v, have)
        assert hit, f'plan step {v} unreachable'
        steps.append(hit)
        have[v] = None
    used = {x for _, a, b in steps for x in (a, b)}
    keep = tuple(r for r, v in enumerate(st.reg) if v in used)
    for op, a, b in steps:
        V = st.vals()
        st.emit(op, V[a], V[b], keep=keep)


def intermediates(V, limit):
    M, vs = {}, list(V)
    for a in range(len(vs)):
        for b in range(len(vs)):
            x, y = vs[a], vs[b]
            for op in ('+', '*') if a <= b else ('-', '/'):
                m = ap(op, x, y)
                if m and m <= limit and m not in V and m not in M:
                    M[m] = (op, x, y)
            if a > b:  # reversed - and / too
                for op in ('-', '/'):
                    m = ap(op, y, x)
                    if m and m <= limit and m not in V and m not in M:
                        M[m] = (op, y, x)
    if vs and 1 not in V:  # x/x = 1; must match combine() so Tiers and M agree
        M[1] = ('/', vs[0], vs[0])
    return M


def partners(t, V):
    """Values m such that t is one op from m and V∪{m}."""
    c = {t // 2} if t % 2 == 0 else set()
    r = int(t ** 0.5)
    for s in (r - 1, r, r + 1):
        if s > 0 and s * s == t: c.add(s)
    for z in V:
        c.add(t - z); c.add(z - t); c.add(t + z); c.add(t * z)
        if t % z == 0: c.add(t // z)
        if z % t == 0: c.add(z // t)
    return c


def chain(t, V, M, limit, beam=12, depth=40):
    """Backward beam search: t = (((base*z1 ± y1)*z2 ± y2)...), z,y from register values V.
    base is in V (0 ops), in M (1 op), or two ops away via M.  Returns (cost, base, moves) or None."""
    Vs = [v for v in V]
    zs = [z for z in Vs if z > 1]
    ys = [0] + Vs

    def term(u):
        if u in V: return 0
        if u in M: return 1
        if partners(u, V) & M.keys(): return 2
        return None

    best = None
    c0 = term(t)
    if c0 is not None:
        return c0, t, []
    frontier = [(0, t, [])]
    for _ in range(depth):
        nxt = {}
        for cost, u, mv in frontier:
            for y in ys:
                for sgn in (-1, 1):
                    s = u + sgn * y
                    if s <= 0 or s > limit or (y == 0 and sgn == 1): continue
                    step = 1 if y == 0 else 2
                    for z in zs:
                        if s % z: continue
                        w = s // z
                        c = cost + step
                        k = term(w)
                        if k is not None and (best is None or c + k < best[0]):
                            best = (c + k, w, mv + [(sgn, y, z)])
                        if w not in nxt or nxt[w][0] > c:
                            nxt[w] = (c, w, mv + [(sgn, y, z)])
        if not nxt: break
        # rank: cost so far + ~1 op per 3 bits still to remove
        frontier = sorted(nxt.values(), key=lambda s: s[0] + s[1].bit_length() / 3)[:beam]
        if best and min(s[0] for s in frontier) >= best[0]: break
    return best


def emit_chain(st, base, moves):
    need = {v for _, y, z in moves for v in (y, z)}
    keep = tuple(r for r, v in enumerate(st.reg) if v in need)  # chain operands must survive eviction
    V = st.vals()
    if base not in V:  # build base in 1 or 2 ops
        hit = one_step(base, V)
        if not hit:
            M = intermediates(V, st.limit)
            m = next(m for m in partners(base, V) if m in M)
            plan = one_step(base, {**V, m: None})
            keep2 = keep + tuple(r for r, v in enumerate(st.reg) if v in plan[1:])
            op, x, y = M[m]
            st.emit(op, V[x], V[y], keep=keep2); V = st.vals()
            hit = one_step(base, V)
        op, x, y = hit
        st.emit(op, V[x], V[y], keep=keep); V = st.vals()
    acc = None
    for sgn, y, z in reversed(moves):  # undo: s = w*z ; u = s - sgn*y
        src = V[base] if acc is None else acc
        acc = st.emit('*', src, V[z], acc, keep=keep)
        if y:
            acc = st.emit('-' if sgn == 1 else '+', acc, V[y], acc, keep=keep)


def build_from(st, t):
    """Fallback: t = v + d, d built by double-and-add from 1."""
    V = st.vals()
    v = max((x for x in V if x < t), default=None)
    if 1 not in V:
        st.emit('/', 0, 0); V = st.vals()
    one = V[1]
    if v is None:  # every known value exceeds t (or t == 1): build t itself from 1
        v, d = None, t
    else:
        d = t - v
    if d == 1:
        st.emit('+', V[v], one); return
    bits = bin(d)[3:]
    acc = st.emit('+', one, one, keep=(one, V.get(v, one)))
    if bits[0] == '1': st.emit('+', acc, one, acc)
    for bit in bits[1:]:
        st.emit('+', acc, acc, acc)
        if bit == '1': st.emit('+', acc, one, acc)
    if v is not None:
        st.emit('+', acc, st.vals()[v] if st.reg[V[v]] != v else V[v])


CFG = dict(beam=12, tiers=5)  # tiers: max pattern cost accepted before falling back to the chain (0 = off)


def mitm_plan(need, V, limit, max_cost):
    """Cheapest target within 4 ops; else the smallest within 5; else the smallest within max_cost."""
    from mitm import Levels
    lv = Levels(V, limit)
    best = None
    for t in need:
        r = lv.best(t, 4)
        if r and (best is None or r[0] < best[0]): best = r
    # ponytail: costs >= 5 are tried on the few smallest targets only; widen if runtime allows
    for c in range(5, max_cost + 1):
        if best: break
        for t in sorted(need)[:3 if c == 5 else 1]:
            best = lv.best(t, c)
            if best: break
    return best[1] if best else None


def tier_plan(need, V, M, limit, max_cost):
    """Cheapest target reachable by a tier pattern (cost 3..max_cost) as a plan, or None."""
    T = Tiers(V, limit)
    best = None
    for t in sorted(need):
        r = T.pair(t)
        if r and (best is None or r[0] < best[0]):
            best = (r[0], t, 'pair', r)
            if r[0] == 3: break
    if best is None or best[0] > 3:
        for t in sorted(need):
            r = T.affine(t)
            if r and (best is None or r[0] < best[0]):
                best = (r[0], t, 'affine', r)
                if r[0] == 3: break
    if (best is None or best[0] > 5) and max_cost >= 5:
        for t in sorted(need):  # T2⊕T2 is costly: stop at the first (smallest) target it reaches
            r = T.pair(t, 5)
            if r:
                best = (r[0], t, 'pair', r); break
    if best is None or best[0] > max_cost:
        return None
    _, t, kind, r = best
    if kind == 'pair':
        _, a, b = r
        return plan_of(a, V, M) + plan_of(b, V, M) + [t]
    _, A, B, C = r
    return plan_of(B, V, M) + plan_of(C, V, M) + [A * B, t]


def solve(start, targets, limit, cfg=CFG):
    st = S(start, targets, limit)
    while st.need:
        V = st.vals()
        hit = None
        for t in sorted(st.need):
            hit = one_step(t, V)
            if hit: break
        if hit:
            op, x, y = hit
            st.emit(op, V[x], V[y]); continue
        M = intermediates(V, limit)
        cnt = {}
        for t in st.need:
            for m in partners(t, V) & M.keys():
                cnt[m] = cnt.get(m, 0) + 1
        if cnt:
            m = max(cnt, key=lambda k: (cnt[k], -k))
            op, x, y = M[m]
            st.emit(op, V[x], V[y]); continue
        if cfg.get('mitm'):
            plan = mitm_plan(st.need, V, limit, cfg['mitm'])
            if plan:
                emit_plan(st, plan); continue
        elif cfg['tiers']:
            plan = tier_plan(st.need, V, M, limit, cfg['tiers'])
            if plan:
                emit_plan(st, plan); continue
        t = min(st.need)
        c = chain(t, V, M, limit, beam=cfg['beam'])
        if c:
            emit_chain(st, c[1], c[2])
            assert t not in st.need, 'chain did not produce target'
        elif 1 not in V:
            st.emit('/', 0, 0)
        elif 2 not in V:
            st.emit('+', V[1], V[1])
        else:
            build_from(st, t)
    return st.prog


def job(args):
    pid, (start, targets, limit), cfg = args
    try:
        prog = solve(start, targets, limit, cfg)
        err = check(prog, start, targets, limit)
    except Exception as e:
        prog, err = None, repr(e)
    return pid, prog, err


if __name__ == '__main__':
    P = load_puzzles(sys.argv[1])
    items = list(P.items())
    cfg = dict(CFG)
    for a in sys.argv[3:]:  # key=value overrides, e.g. beam=24 tiers=0 limit=20
        k, v = a.split('=')
        cfg[k] = int(v)
    if 'limit' in cfg:
        items = items[:cfg.pop('limit')]
    if 'skip_tiny' in cfg and cfg.pop('skip_tiny'):
        items = [it for it in items if it[1][2] > 100]
    items = [(pid, p, cfg) for pid, p in items]
    print(cfg)
    with Pool(cfg.pop('procs', 12)) as pool:  # 16 workers x big levels exhausts RAM
        res = list(pool.imap_unordered(job, items, chunksize=1))
    bad = [(pid, err) for pid, _, err in res if err]
    for pid, err in bad[:10]: print('FAIL', pid, err)
    ok = [(pid, prog) for pid, prog, err in res if not err]
    print(f'solved {len(ok)}/{len(res)}  instructions {sum(len(p) for _, p in ok)}')
    with open(sys.argv[2], 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'program'])
        for pid, prog in ok:
            w.writerow([pid, json.dumps(prog, separators=(',', ':'))])
