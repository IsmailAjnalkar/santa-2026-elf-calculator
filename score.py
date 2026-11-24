"""Strict local replica of the Santa 2026 scorer.  usage: python score.py submission.csv [puzzles.csv]"""
import csv, json, sys

NREG, MAX_LEN, MAX_BYTES = 48, 4096, 131072
OPS = ('+', '-', '*', '/')


def load_puzzles(path):
    out = {}
    for r in csv.DictReader(open(path, newline='')):
        out[r['id']] = (json.loads(r['starting_values']), json.loads(r['targets']), int(r['value_limit']))
    return out


def check(prog, start, targets, limit):
    """None if prog is legal and covers every target, else an error string."""
    if not isinstance(prog, list) or len(prog) > MAX_LEN:
        return 'not a list or too long'
    reg = [None] * NREG
    reg[:len(start)] = start
    need = set(targets) - set(start)
    for k, ins in enumerate(prog):
        if not (isinstance(ins, list) and len(ins) == 4 and ins[0] in OPS):
            return f'{k}: malformed {ins}'
        op, a, b, d = ins
        if any(type(r) is not int or not 0 <= r < NREG for r in (a, b, d)):
            return f'{k}: bad register {ins}'
        x, y = reg[a], reg[b]
        if x is None or y is None:
            return f'{k}: undefined source {ins}'
        if op == '+': v = x + y
        elif op == '-': v = x - y
        elif op == '*': v = x * y
        elif x % y: return f'{k}: inexact division {ins}'
        else: v = x // y
        if not 0 < v <= limit:
            return f'{k}: value {v} out of range {ins}'
        reg[d] = v
        need.discard(v)
    return f'{len(need)} targets missing' if need else None


def score(sub_path, puz_path):
    P = load_puzzles(puz_path)
    rows = list(csv.reader(open(sub_path, newline='', encoding='utf-8')))
    assert rows[0] == ['id', 'program'], f'bad header {rows[0]}'
    seen, total = set(), 0
    for pid, cell in rows[1:]:
        assert pid in P and pid not in seen, f'bad/duplicate id {pid}'
        seen.add(pid)
        assert len(cell.encode()) <= MAX_BYTES, f'{pid}: cell too big'
        prog = json.loads(cell)
        err = check(prog, *P[pid])
        assert err is None, f'{pid}: {err}'
        total += len(prog)
    assert seen == set(P), f'missing ids: {len(set(P) - seen)}'
    return total


# Worked example from the Data page must pass; a wrong variant must fail.
assert check([["/", 0, 0, 1], ["-", 0, 1, 2], ["+", 0, 2, 3], ["+", 0, 3, 4]], [3], [1, 2, 5, 8], 9) is None
assert check([["/", 0, 0, 1], ["-", 1, 0, 2]], [3], [1], 9) is not None

if __name__ == '__main__':
    print(score(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'data/puzzles.csv'))
