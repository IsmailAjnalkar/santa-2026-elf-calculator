"""Kaggle-side LNS run for puzzles 501-1000 (private notebook).
Builds the public elfopt engine, improves our best programs for scored_0501..1000 in adaptive rounds,
validates every returned program against the official rules, and writes /kaggle/working/submission.csv
(all 1,000 puzzles; 1-500 untouched) after every round. Never submits."""
import csv, glob, json, os, random, subprocess, time

T_END = time.time() + 11 * 3600          # stay inside Kaggle's 12 h limit
NREG, LO, HI = 48, 501, 1000
csv.field_size_limit(1 << 26)


def find(name):
    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True))
    assert hits, f'{name} not found under /kaggle/input'
    return hits[0]


def check(prog, start, targets, limit):
    """None if prog is legal and covers every target (official rules), else an error string."""
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
assert all(check(S[k], *P[k]) is None for k in P), 'input best.csv is not fully valid'
base = sum(len(p) for p in S.values())

EXE = '/kaggle/working/elfopt'
subprocess.run(['g++', '-std=c++17', '-O3', '-march=native', '-o', EXE, find('elfopt.cpp')], check=True)


def encode(pid, prog):
    s, t, L = P[pid]
    head = ['P', pid, str(len(s)), *map(str, s), str(len(t)), *map(str, t), str(L), '48', str(len(prog))]
    return ' '.join(head) + '\n' + ''.join(f'{o} {i} {j} {d}\n' for o, i, j, d in prog)


def parse(text):
    res, lines, k = {}, text.split('\n'), 0
    while k < len(lines):
        f = lines[k].split()
        if f and f[0] == 'R':
            n = int(f[3])
            res[f[1]] = [[o, int(i), int(j), int(d)] for o, i, j, d in (lines[k + 1 + q].split() for q in range(n))]
            k += n + 1
        else:
            k += 1
    return res


def save():
    with open('/kaggle/working/submission.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['id', 'program'])
        for pid in sorted(S): w.writerow([pid, json.dumps(S[pid], separators=(',', ':'))])


jobs, tper, success, rnd = os.cpu_count() or 4, 30, {}, 0
log = open('/kaggle/working/log.txt', 'w')
save()
while time.time() < T_END - 600:
    t0 = time.time()
    ids = [k for k in P if LO <= int(k.split('_')[1]) <= HI]
    random.shuffle(ids); ids.sort(key=lambda k: (-success.get(k, 0), -len(S[k])))
    left = T_END - 600 - time.time()
    ids = ids[:max(jobs, int(left / tper) * jobs)]           # fit the round into the time left
    procs = []
    for i in range(jobs):
        sh = ids[i::jobs]
        if not sh: continue
        open(f'/kaggle/working/in_{i}.txt', 'w').write(''.join(encode(k, S[k]) for k in sh))
        procs.append((subprocess.Popen([EXE, '--time', str(tper), '--seed', str(random.randrange(1, 10**9)),
                                        '--kicks', '1000', '--kslack', '2'],
                                       stdin=open(f'/kaggle/working/in_{i}.txt'),
                                       stdout=open(f'/kaggle/working/out_{i}.txt', 'w'),
                                       stderr=subprocess.DEVNULL), i))
    for p, _ in procs: p.wait()
    gained = 0
    for _, i in procs:
        for pid, prog in parse(open(f'/kaggle/working/out_{i}.txt').read()).items():
            if pid in P and LO <= int(pid.split('_')[1]) <= HI and len(prog) < len(S[pid]) and check(prog, *P[pid]) is None:
                gained += len(S[pid]) - len(prog); S[pid] = prog; success[pid] = success.get(pid, 0) + 1
                log.write(f'{pid} -> {len(prog)}\n')
    save()
    tper = max(30, round(tper / 1.5)) if gained else min(240, tper * 2)   # adaptive budget
    msg = f'round {rnd}: {len(ids)} puzzles, saved {gained}, total {sum(len(p) for p in S.values())} (start {base}), {round(time.time() - t0)}s, next {tper}s'
    print(msg, flush=True); log.write(msg + '\n'); log.flush()
    rnd += 1
for f in glob.glob('/kaggle/working/*_*.txt'): os.remove(f)
print('done', sum(len(p) for p in S.values()), 'from', base)
