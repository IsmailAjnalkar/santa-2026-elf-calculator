"""Overnight LNS loop around the public C++ engine (bin/elfopt.exe, built from external/dag-lns/elfopt.cpp).
usage: python lns_driver.py --until 08:30 [--time 20] [--jobs 12] [--min 60 --max 180] [--no-submit]
Each round: shard out/best.csv -> engine processes with fresh seeds -> score.check every returned program ->
keep strictly shorter -> rewrite out/best.csv (re-merged with whatever is there now) -> submit if the total dropped."""
import argparse, csv, datetime, json, os, random, shutil, subprocess, sys, time
from score import load_puzzles, check, score

csv.field_size_limit(1 << 26)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--until', default='08:30'); ap.add_argument('--time', type=float, default=20)
ap.add_argument('--jobs', type=int, default=12); ap.add_argument('--min', type=int, default=60)
ap.add_argument('--max', type=int, default=180); ap.add_argument('--no-submit', action='store_true')
ap.add_argument('--best', default='out/best.csv')  # file to optimise in place
ap.add_argument('--ids', default='')  # e.g. 1-500: only these scored_NNNN puzzles
a = ap.parse_args()

P = load_puzzles('data/puzzles.csv')
BEST, EXE, WORK = a.best, os.path.abspath('bin/elfopt.exe'), 'out/lns_work_' + os.path.basename(a.best).split('.')[0]
os.makedirs(WORK, exist_ok=True)
log = open('out/lns.log', 'a', encoding='utf-8')
def say(m):
    m = f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {m}'; print(m, flush=True); log.write(m + '\n'); log.flush()

hh, mm = map(int, a.until.split(':'))
stop = datetime.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
if stop <= datetime.datetime.now(): stop += datetime.timedelta(days=1)


def read_best():
    return {r['id']: json.loads(r['program']) for r in csv.DictReader(open(BEST, newline='', encoding='utf-8'))}


def write_best(S):
    tmp = BEST + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['id', 'program'])
        for pid in sorted(S): w.writerow([pid, json.dumps(S[pid], separators=(',', ':'))])
    for _ in range(20):  # Windows refuses the replace while another process has best.csv open
        try:
            os.replace(tmp, BEST); return
        except PermissionError:
            time.sleep(15)
    say('could not replace best.csv for 5 min; gains kept in best.csv.tmp')


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


STATE, JOURNAL, MILESTONES = 'out/opt_state.json', 'out/improvements.csv', (1, 10, 25, 50, 100, 250, 500)
st = json.load(open(STATE)) if os.path.exists(STATE) else {}
st.setdefault('baseline', score(BEST, 'data/puzzles.csv')); st.setdefault('milestones', [])
st.setdefault('time', a.time); st.setdefault('success', {})
save_state = lambda: json.dump(st, open(STATE, 'w'), indent=1)
save_state()
if not os.path.exists(JOURNAL):
    open(JOURNAL, 'w').write('time,puzzle,old,new,saved,algorithm,round_seconds\n')
os.makedirs('out/milestones', exist_ok=True)

say(f'start: until {stop:%Y-%m-%d %H:%M}, time/puzzle {st["time"]}s (adaptive), jobs {a.jobs}, '
    f'lengths {a.min}-{a.max}, baseline {st["baseline"]}')
rnd = 0
while datetime.datetime.now() < stop:
    t_round = time.time()
    S = read_best()
    lo, hi = map(int, a.ids.split('-')) if a.ids else (1, 10**9)
    ids = [k for k in P if a.min <= len(S[k]) <= a.max and lo <= int(k.split('_')[1]) <= hi]
    # puzzles that improved before first, then longer programs; random tie-break
    random.shuffle(ids)
    ids.sort(key=lambda k: (-st['success'].get(k, 0), -len(S[k])))
    a.time = st['time']
    # cap round length so the loop checks the clock: ~ (puzzles per shard) * time <= 45 min
    per_shard = max(1, int(45 * 60 / a.time))
    ids = ids[:per_shard * a.jobs]
    shards = [ids[i::a.jobs] for i in range(a.jobs)]
    procs = []
    for i, sh in enumerate(shards):
        if not sh: continue
        fin = os.path.join(WORK, f'in_{i:02d}.txt')
        open(fin, 'w').write(''.join(encode(k, S[k]) for k in sh))
        cmd = [EXE, '--time', str(a.time), '--seed', str(random.randrange(1, 10**9)), '--kicks', '1000', '--kslack', '2']
        procs.append((subprocess.Popen(cmd, stdin=open(fin), stdout=open(os.path.join(WORK, f'out_{i:02d}.txt'), 'w'),
                                       stderr=open(os.path.join(WORK, f'log_{i:02d}.txt'), 'w')), i))
    for p, _ in procs: p.wait()
    gained = bad = 0
    cur = read_best()  # re-read in case refresh.py improved it meanwhile
    secs = round(time.time() - t_round)
    jl = open(JOURNAL, 'a')
    for _, i in procs:
        for pid, prog in parse(open(os.path.join(WORK, f'out_{i:02d}.txt')).read()).items():
            if pid not in P: continue
            if check(prog, *P[pid]) is not None: bad += 1; continue
            if len(prog) < len(cur[pid]):
                jl.write(f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S},{pid},{len(cur[pid])},{len(prog)},'
                         f'{len(cur[pid]) - len(prog)},LNS elfopt t={a.time}s,{secs}\n')
                st['success'][pid] = st['success'].get(pid, 0) + 1
                gained += len(cur[pid]) - len(prog); cur[pid] = prog
    jl.close()
    backup = BEST + '.bak'
    shutil.copy(BEST, backup)
    write_best(cur)
    try:
        tot = score(BEST, 'data/puzzles.csv')  # full independent validation of all 1,000 programs
    except AssertionError as e:  # never keep an invalid file: roll back
        shutil.copy(backup, BEST); say(f'round {rnd}: validation failed ({e}); rolled back'); continue
    # adaptive budget: spend longer per puzzle while rounds come up empty, shorten again after a gain
    st['time'] = max(30, round(st['time'] / 1.5)) if gained else min(240, st['time'] * 2)
    saved_total = st['baseline'] - tot
    for m in MILESTONES:
        if saved_total >= m and m not in st['milestones']:
            shutil.copy(BEST, f'out/milestones/submission_saved_{m}.csv'); st['milestones'].append(m)
            say(f'milestone: {m} instructions saved since baseline -> out/milestones/submission_saved_{m}.csv')
    save_state()
    say(f'round {rnd}: {len(ids)} puzzles, saved {gained}, invalid {bad}, total {tot}, '
        f'{secs}s, next time/puzzle {st["time"]}s, saved since baseline {saved_total}')
    if gained > 0 and not a.no_submit:  # only its own gains; refresh.py submits public ones
        from refresh import submit, last_submitted as last_ok   # shared out/last_submitted.txt record
        if last_ok() is None or tot < last_ok():
            submit(tot, 'LNS', say)
    rnd += 1
say('stop time reached')
