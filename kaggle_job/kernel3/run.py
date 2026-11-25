"""Kaggle-side run of Wang Jakie's public C++ strategies (Apache-2.0; anneal / incremental / compact_rotate)
on all 1,000 puzzles, starting from our best programs. Every returned program is replayed against the official
rules; only strictly shorter valid ones are kept. Writes /kaggle/working/submission.csv after every pass. Never submits."""
import csv, glob, json, os, random, subprocess, time
from concurrent.futures import ThreadPoolExecutor

T_END = time.time() + 11 * 3600
NREG = 48
csv.field_size_limit(1 << 26)


def find(name):
    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True))
    assert hits, f'{name} not found'
    return hits[0]


def check(prog, start, targets, limit):
    if not isinstance(prog, list) or len(prog) > 4096: return 'too long'
    reg = [None] * NREG; reg[:len(start)] = start
    need = set(targets) - set(start)
    for ins in prog:
        if not (isinstance(ins, list) and len(ins) == 4 and ins[0] in '+-*/'): return 'malformed'
        op, a, b, d = ins
        if any(type(r) is not int or not 0 <= r < NREG for r in (a, b, d)): return 'register'
        x, y = reg[a], reg[b]
        if x is None or y is None: return 'undefined'
        if op == '+': v = x + y
        elif op == '-': v = x - y
        elif op == '*': v = x * y
        elif x % y: return 'inexact'
        else: v = x // y
        if not 0 < v <= limit: return 'range'
        reg[d] = v; need.discard(v)
    return 'missing' if need else None


P = {r['id']: (json.loads(r['starting_values']), json.loads(r['targets']), int(r['value_limit']))
     for r in csv.DictReader(open(find('puzzles.csv')))}
S = {r['id']: json.loads(r['program']) for r in csv.DictReader(open(find('best.csv'), encoding='utf-8'))}
assert all(check(S[k], *P[k]) is None for k in P)
base = sum(len(p) for p in S.values())

src = os.path.dirname(find('anneal.cpp'))
EXE = {}
for name in ('anneal', 'incremental', 'compact_rotate'):
    EXE[name] = f'/kaggle/working/{name}'
    subprocess.run(['g++', '-O3', '-std=c++17', '-march=native', '-I', src, '-o', EXE[name], f'{src}/{name}.cpp'], check=True)


def job_text(k, rounds, seed):
    s, t, L = P[k]; u = sorted(set(t)); prog = S[k]
    rows = [f'{L} {NREG} {rounds} 0 {seed}', ' '.join(map(str, (len(s), *s))), ' '.join(map(str, (len(u), *u))), '1',
            str(len(prog))] + [f'{o} {a} {b} {d}' for o, a, b, d in prog]
    return '\n'.join(rows)


def run(name, rounds, batch):
    text = '\n'.join([str(len(batch))] + [job_text(k, rounds, random.randrange(1, 2**31)) for k in batch]) + '\n'
    r = subprocess.run([EXE[name]], input=text, capture_output=True, text=True)
    out = {}
    for k, line in zip(batch, r.stdout.splitlines()):
        tok = line.split()
        if not tok: continue
        prog = [[tok[i], int(tok[i + 1]), int(tok[i + 2]), int(tok[i + 3])] for i in range(1, len(tok), 4)]
        if len(prog) < len(S[k]) and check(prog, *P[k]) is None: out[k] = prog
    return out


def save():
    with open('/kaggle/working/submission.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['id', 'program'])
        for pid in sorted(S): w.writerow([pid, json.dumps(S[pid], separators=(',', ':'))])


log = open('/kaggle/working/log.txt', 'w'); save()
schedule = [('anneal', 1000), ('incremental', 700), ('compact_rotate', 300), ('anneal', 700)]
jobs, it = os.cpu_count() or 4, 0
while time.time() < T_END - 900:
    name, rounds = schedule[it % len(schedule)]; it += 1
    t0 = time.time(); ids = sorted(P, key=lambda k: -len(S[k]))
    batches = [ids[i:i + 4] for i in range(0, len(ids), 4)]
    gain = 0
    with ThreadPoolExecutor(jobs) as pool:
        futs = []
        for b in batches:
            if time.time() > T_END - 900: break
            futs.append(pool.submit(run, name, rounds, b))
        for f in futs:
            for k, prog in f.result().items():
                if len(prog) < len(S[k]): gain += len(S[k]) - len(prog); S[k] = prog
    save()
    msg = f'pass {it} {name} rounds={rounds}: saved {gain}, total {sum(len(p) for p in S.values())} (start {base}), {round(time.time() - t0)}s'
    print(msg, flush=True); log.write(msg + '\n'); log.flush()
print('done', sum(len(p) for p in S.values()), 'from', base)
