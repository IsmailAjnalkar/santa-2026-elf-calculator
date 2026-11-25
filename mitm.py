"""Numba meet-in-the-middle over values 1..3 instructions from the register pool.
Level k holds sorted unique values with provenance (parent index in level k-1, op code, base operand index)."""
import numpy as np
from numba import njit

# op codes: v = parent op base, or base op parent; 6/7 use the parent twice
# 0 p+b  1 p-b  2 b-p  3 p*b  4 p/b  5 b/p  6 p+p  7 p*p


@njit(cache=True)
def _apply(code, p, b, limit):
    if code == 0: v = p + b
    elif code == 1: v = p - b
    elif code == 2: v = b - p
    elif code == 3: v = p * b if p <= limit // b else 0
    elif code == 4: v = p // b if p % b == 0 else 0
    elif code == 5: v = b // p if b % p == 0 else 0
    elif code == 6: v = p + p
    else: v = p * p if p <= limit // p else 0
    return v if 0 < v <= limit else 0


@njit(cache=True)
def _expand(par, base, limit, with_self):
    n = 0
    for i in range(len(par)):
        for code in range(6):
            for j in range(len(base)):
                if _apply(code, par[i], base[j], limit): n += 1
        if with_self:
            for code in (6, 7):
                if _apply(code, par[i], par[i], limit): n += 1
    val = np.empty(n, np.int64); pi = np.empty(n, np.int32); op = np.empty(n, np.int8); bj = np.empty(n, np.int16)
    k = 0
    for i in range(len(par)):
        for code in range(6):
            for j in range(len(base)):
                v = _apply(code, par[i], base[j], limit)
                if v:
                    val[k] = v; pi[k] = i; op[k] = code; bj[k] = j; k += 1
        if with_self:
            for code in (6, 7):
                v = _apply(code, par[i], par[i], limit)
                if v:
                    val[k] = v; pi[k] = i; op[k] = code; bj[k] = -1; k += 1
    return val, pi, op, bj


@njit(cache=True)
def _count(par, base, limit):
    n = 0
    for i in range(len(par)):
        for code in range(6):
            for j in range(len(base)):
                if _apply(code, par[i], base[j], limit): n += 1
    return n


@njit(cache=True)
def _has(B, x):
    i = np.searchsorted(B, x)
    return i < len(B) and B[i] == x


@njit(cache=True)
def find_pair(t, A, B, limit):
    """Index pair (i, j) with A[i] op B[j] == t, or (-1, -1)."""
    for i in range(len(A)):
        a = A[i]
        for r in range(6):
            if r == 0: b = t - a if a < t else 0                      # t = a + b
            elif r == 1: b = a - t if a > t else 0                    # t = a - b
            elif r == 2: b = t + a if a <= limit - t else 0           # t = b - a
            elif r == 3: b = t // a if t % a == 0 else 0              # t = a * b
            elif r == 4: b = a // t if a % t == 0 else 0              # t = a / b
            else: b = t * a if a <= limit // t else 0                 # t = b / a
            if b > 0:
                j = np.searchsorted(B, b)
                if j < len(B) and B[j] == b: return i, j
    return -1, -1


@njit(cache=True)
def find_affine(t, A, B, C, limit):
    """(i, j, k, sign) with A[i]*B[j] + sign*C[k] == t, or i == -1.  Scans, for each a, only the b with |t - a*b| <= max(C)."""
    if len(C) == 0: return -1, -1, -1, 0
    cmax = C[-1]
    for i in range(len(A)):
        a = A[i]
        if a < 2: continue
        lo = (t - cmax) // a if t > cmax else 1
        hi = (t + cmax) // a + 1
        j = np.searchsorted(B, lo)
        while j < len(B) and B[j] <= hi:
            b = B[j]
            if b <= limit // a:
                p = a * b
                if p != t:
                    c = t - p if p < t else p - t
                    k = np.searchsorted(C, c)
                    if k < len(C) and C[k] == c:
                        return i, j, k, (1 if p < t else -1)
            j += 1
    return -1, -1, -1, 0


class Levels:
    def __init__(self, V, limit, cap3=3_000_000, depth=3):
        self.limit = limit
        base = np.array(sorted(V), np.int64)
        self.L = [base]
        self.prov = [None]
        seen = base
        for k in range(1, depth + 1):
            par = self.L[-1]
            if k == 3 and _count(par, base, limit) > cap3:
                break
            val, pi, op, bj = _expand(par, base, limit, k == 2)
            u, idx = np.unique(val, return_index=True)
            keep = ~np.isin(u, seen)
            u, idx = u[keep], idx[keep]
            self.L.append(u)
            self.prov.append((pi[idx], op[idx], bj[idx]))
            seen = np.union1d(seen, u)

    def level_of(self, v):
        for k, A in enumerate(self.L):
            if _has(A, v): return k
        return None

    def build(self, v):
        """Plan (values in creation order, each one op from the pool and earlier ones) for a level value."""
        k = self.level_of(v)
        plan = []
        i = int(np.searchsorted(self.L[k], v))
        while k > 0:
            plan.append(int(self.L[k][i]))
            i = int(self.prov[k][0][i]); k -= 1
        return plan[::-1]

    def best(self, t, max_cost, affine_a_max=1):
        """(cost, plan) for the cheapest pattern up to max_cost, or None."""
        lk = self.level_of(t)
        if lk is not None and lk <= max_cost:
            return lk, self.build(t)
        L, n = self.L, len(self.L) - 1
        # pair: t = a op b (cost i+j+1);  affine: t = a*b ± c (cost i+j+k+2, a,b from levels <= 2)
        combos = [(i + j + 1, 'pair', i, j, 0) for i in range(n + 1) for j in range(i, n + 1)
                  if not (i == 0 and j <= n - 1)]  # L0 ⊕ Lj is level j+1, already checked
        # a from levels 0-1, b from 0-2: an L2 x L2 scan is ~1e10 steps when c can be large
        combos += [(i + j + k + 2, "affine", i, j, k) for i in range(min(n, affine_a_max) + 1) for j in range(i, min(n, 2) + 1)
                   for k in range(n + 1) if i + j > 0]
        combos.sort(key=lambda c: (c[0], c[1] == 'affine'))
        for c, kind, i, j, k in combos:
            if c > max_cost: break
            if kind == 'pair':
                A, B = (L[i], L[j]) if len(L[i]) <= len(L[j]) else (L[j], L[i])
                a, b = find_pair(t, A, B, self.limit)
                if a >= 0:
                    va, vb = int(A[a]), int(B[b])
                    return c, self.build(va) + self.build(vb) + [t]
            else:
                a, b, ci, sgn = find_affine(t, L[i], L[j], L[k], self.limit)
                if a >= 0:
                    va, vb, vc = int(L[i][a]), int(L[j][b]), int(L[k][ci])
                    return c, self.build(va) + self.build(vb) + self.build(vc) + [va * vb, t]
        return None
