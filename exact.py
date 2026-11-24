"""Optimal programs for small puzzles by iterative-deepening DFS.  usage: python exact.py puzzles.csv out.csv [max_limit]
Every new value gets its own register (fine while len(start) + depth <= 48)."""
import csv, json, sys, time
from score import load_puzzles, check, NREG

OPS = ('+', '-', '*', '/')


def ap(op, x, y):
    if op == '+': return x + y
    if op == '-': return x - y if x > y else 0
    if op == '*': return x * y
    return x // y if x % y == 0 else 0


def solve_exact(start, targets, limit, max_depth=12, budget=20.0):
    need0 = frozenset(targets) - set(start)
    deadline = time.time() + budget

    def dfs(vals, need, depth, path, seen):
        if not need: return path
        if len(need) > depth or time.time() > deadline: return None
        key = (frozenset(vals), depth)
        if key in seen: return None
        seen.add(key)
        vs = list(vals)
        for i, x in enumerate(vs):
            for y in vs[i:]:
                for a, b in ((x, y), (y, x)):
                    for op in OPS:
                        v = ap(op, a, b)
                        if 0 < v <= limit and v not in vals:
                            r = dfs(vals | {v}, need - {v}, depth - 1, path + [(op, a, b, v)], seen)
                            if r is not None: return r
        return None

    for d in range(len(need0), min(max_depth, NREG - len(start)) + 1):
        r = dfs(frozenset(start), need0, d, [], set())
        if r is not None: return r
        if time.time() > deadline: return None
    return None


def to_program(start, path):
    reg = {v: i for i, v in enumerate(start)}
    prog, nxt = [], len(start)
    for op, a, b, v in path:
        prog.append([op, reg[a], reg[b], nxt])
        reg.setdefault(v, nxt); nxt += 1
    return prog


assert len(solve_exact([3], [1, 2, 5, 8], 9)) == 4  # worked example from the Data page

if __name__ == '__main__':
    max_limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    P = load_puzzles(sys.argv[1])
    rows = []
    for pid, (s, t, L) in P.items():
        if L > max_limit: continue
        path = solve_exact(s, t, L)
        if path is None: print('timeout', pid); continue
        prog = to_program(s, path)
        assert check(prog, s, t, L) is None, pid
        rows.append((pid, prog))
    with open(sys.argv[2], 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['id', 'program'])
        for pid, prog in rows: w.writerow([pid, json.dumps(prog, separators=(',', ':'))])
    print(f'{len(rows)} puzzles, {sum(len(p) for _, p in rows)} instructions')
