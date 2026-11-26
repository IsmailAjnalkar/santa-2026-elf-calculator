"""Refresh: pull public notebook CSV outputs, merge with our best, submit when best beats the last confirmed submission.
usage: python refresh.py [--no-submit] [--full]     (run from the santa-2026 folder; safe to run every few minutes)
Only notebooks re-run since the last pass are downloaded (--full re-downloads all). A lock file stops overlapping runs.
Downloaded files are data only: merge.py re-validates every program with score.check."""
import csv, datetime, hashlib, io, json, os, re, shutil, subprocess, sys, tempfile, time, zipfile

COMP, BEST = 'santa-2026', 'out/best.csv'
LAST, SEEN, LOCK = 'out/last_submitted.txt', 'out/kernel_seen.json', 'out/refresh.lock'
DAILY_CAP = 50          # user: at most 50 submissions per day
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# This network's IPv6 route to Kaggle/Google hosts stalls; force IPv4 and a socket timeout for every CLI call.
IPV4 = ('import socket, sys, urllib3.util.connection as uc; uc.allowed_gai_family = lambda: socket.AF_INET; '
        'socket.setdefaulttimeout(120); sys.argv = ["kaggle"] + sys.argv[1:]; from kaggle.cli import main; main()')


def kaggle(*args, timeout=300):
    env = dict(os.environ, PYTHONIOENCODING='utf-8')  # emoji in notebook titles crash the default Windows codec
    r = subprocess.run([sys.executable, '-c', IPV4, *args], capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=timeout, env=env)
    return r.stdout + (('\n[stderr] ' + r.stderr.strip()) if r.stderr.strip() else '')


def kernels():
    """{ref: lastRunTime} for the most-voted and most recently run notebooks."""
    out = {}
    for sort in ('voteCount', 'dateRun'):
        txt = kaggle('kernels', 'list', '--competition', COMP, '--sort-by', sort, '--page-size', '60', '-v')
        for r in csv.DictReader(io.StringIO(txt.split('\n[stderr]')[0])):
            if r.get('ref'): out[r['ref']] = r.get('lastRunTime', '')
    return out


def last_submitted():
    return int(open(LAST).read()) if os.path.exists(LAST) else None


def used_today():
    m = re.search(r'Submissions today:\s*(\d+)', kaggle('competitions', 'submission-limits', '-c', COMP))
    return int(m.group(1)) if m else 0


def submit(total, label, say, tries=3):
    """Submit out/best.csv; record the score only when Kaggle confirms it. Retries transient failures."""
    if used_today() >= DAILY_CAP:
        say(f'daily cap {DAILY_CAP} reached; {total} will be submitted tomorrow'); return False
    shutil.copy(BEST, 'out/upload.csv')  # upload a copy: Windows locks the file being read, which blocked best.csv writes
    for t in range(tries):
        try:
            r = subprocess.run([sys.executable, 'submit_api.py', 'out/upload.csv', f'{label} (local {total})'],
                               capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
            out = r.stdout + r.stderr[-500:]
        except subprocess.TimeoutExpired:
            out = 'timeout'
        if 'Successfully submitted' in out:
            open(LAST, 'w').write(str(total))
            say(f'submitted {total}: ok'); return True
        time.sleep(20)
    tail = ' '.join(l for l in out.strip().splitlines() if '%|' not in l)[-300:]
    say(f'submitted {total}: FAILED after {tries} tries, will retry next run: {tail or "no output"}')
    return False


def keep_outside_range(rng):
    """--ids lo-hi: candidate keeps out/best.csv's programs for every puzzle outside lo-hi (teammate's puzzles)."""
    from score import score
    lo, hi = map(int, rng.split('-'))
    csv.field_size_limit(1 << 26)
    rd = lambda p: {r['id']: r['program'] for r in csv.DictReader(open(p, newline='', encoding='utf-8'))}
    best, cand = rd(BEST), rd('out/candidate.csv')
    with open('out/candidate.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['id', 'program'])
        for pid in sorted(best):
            w.writerow([pid, cand[pid] if lo <= int(pid.split('_')[1]) <= hi else best[pid]])
    return score('out/candidate.csv', 'data/puzzles.csv')


def main():
    log = open('out/refresh.log', 'a', encoding='utf-8')
    say = lambda m: (print(m), log.write(f'{datetime.datetime.now():%Y-%m-%d %H:%M} {m}\n'), log.flush())
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 30 * 60:
        say('previous refresh still running; skipped'); return
    open(LOCK, 'w').write(str(os.getpid()))
    try:
        run(say)
    finally:
        os.remove(LOCK)


REF_OK = re.compile(r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')


def fetch(ref):
    """Download a notebook's CSV outputs into a fresh temp dir, then move only plain .csv files that
    resolve inside it into public/<owner>_<slug>/. Output file names come from third parties."""
    if not REF_OK.match(ref) or '..' in ref:
        raise ValueError(f'unexpected notebook ref {ref!r}')
    tmp = tempfile.mkdtemp(prefix='santa_nb_')  # system temp dir, outside the project tree
    try:
        kaggle('kernels', 'output', ref, '-p', tmp, '--file-pattern', r'.*\.csv$', '-q', '-o', timeout=90)
        root = os.path.realpath(tmp)
        dest = os.path.join('public', ref.replace('/', '_'))
        os.makedirs(dest, exist_ok=True)
        for dp, _, fs in os.walk(tmp):
            for f in fs:
                p = os.path.realpath(os.path.join(dp, f))
                if f.endswith('.csv') and os.path.isfile(p) and not os.path.islink(os.path.join(dp, f)) \
                        and os.path.commonpath([root, p]) == root:
                    shutil.move(p, os.path.join(dest, os.path.basename(p)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def datasets():
    """{ref: lastUpdated} for recently updated public datasets matching 'santa 2026'."""
    out = {}
    for page in (1, 2):
        txt = kaggle('datasets', 'list', '-s', 'santa 2026', '--sort-by', 'updated', '-p', str(page), '-v')
        for r in csv.DictReader(io.StringIO(txt.split('\n[stderr]')[0])):
            if r.get('ref') and REF_OK.match(r['ref']): out[r['ref']] = r.get('lastUpdated', '')
    return out


def fetch_dataset(ref, max_bytes=50 << 20):
    """Download a dataset's .csv files (each <= max_bytes) one by one into a fresh temp dir; extract zips only after
    checking member names; move plain .csv files that resolve inside the temp dir into public_ds/<owner>_<slug>/."""
    if not REF_OK.match(ref) or '..' in ref:
        raise ValueError(f'unexpected dataset ref {ref!r}')
    listing = kaggle('datasets', 'files', ref, '-v', '--page-size', '200').split('\n[stderr]')[0]
    names = [r['name'] for r in csv.DictReader(io.StringIO(listing))
             if r.get('name', '').endswith('.csv') and int(r.get('totalBytes') or r.get('size') or 0) <= max_bytes]
    dest = os.path.join('public_ds', ref.replace('/', '_'))
    os.makedirs(dest, exist_ok=True)
    for name in names:
        tmp = tempfile.mkdtemp(prefix='santa_ds_')
        try:
            kaggle('datasets', 'download', ref, '-f', name, '-p', tmp, '-q', '-o', timeout=120)
            root = os.path.realpath(tmp)
            for z in [os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith('.zip')]:
                with zipfile.ZipFile(z) as zf:
                    for m in zf.namelist():
                        if m.endswith('.csv') and '..' not in m and not os.path.isabs(m) and ':' not in m:
                            zf.extract(m, tmp)
            for dp, _, fs in os.walk(tmp):
                for f in fs:
                    p = os.path.realpath(os.path.join(dp, f))
                    if f.endswith('.csv') and os.path.isfile(p) and os.path.commonpath([root, p]) == root:
                        safe = re.sub(r'[^A-Za-z0-9_.-]', '_', name.replace('/', '__'))
                        shutil.move(p, os.path.join(dest, safe))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def check_puzzles():
    """data/puzzles.csv must never change: compare with the hash recorded on first use."""
    h = hashlib.sha256(open('data/puzzles.csv', 'rb').read()).hexdigest()
    pin = 'out/puzzles.sha256'
    if not os.path.exists(pin): open(pin, 'w').write(h)
    if open(pin).read().strip() != h:
        raise RuntimeError('data/puzzles.csv changed since it was pinned; refusing to merge')


def run(say):
    check_puzzles()
    seen = json.load(open(SEEN)) if os.path.exists(SEEN) and '--full' not in sys.argv else {}
    ks, changed = kernels(), 0
    for ref, stamp in ks.items():
        if seen.get(ref) == stamp and stamp: continue
        try:
            fetch(ref)  # slow ones retry next run
            seen[ref] = stamp; changed += 1
        except subprocess.TimeoutExpired:
            say(f'timeout {ref}')
        except ValueError as e:
            say(f'skipped: {e}')
    for ref, stamp in datasets().items():   # public datasets also carry frontier programs
        key = 'ds:' + ref
        if seen.get(key) == stamp and stamp: continue
        try:
            fetch_dataset(ref)
            seen[key] = stamp; changed += 1
        except subprocess.TimeoutExpired:
            say(f'timeout dataset {ref}')
        except (ValueError, zipfile.BadZipFile) as e:
            say(f'skipped dataset: {e}')
    json.dump(seen, open(SEEN, 'w'))
    check_puzzles()
    from score import score
    best_score = score(BEST, 'data/puzzles.csv') if os.path.exists(BEST) else None
    last = last_submitted()
    if changed or best_score is None:
        files = [os.path.join(dp, f) for top in ('public', 'public_ds') for dp, _, fs in os.walk(top) for f in fs if f.endswith('.csv')]
        open('out/merge_files.txt', 'w', encoding='utf-8').write('\n'.join(files))
        r = subprocess.run([sys.executable, 'merge.py', 'out/candidate.csv', 'data/puzzles.csv',
                            *([BEST] if best_score else []), '@out/merge_files.txt'], capture_output=True, text=True)
        lines = r.stdout.strip().splitlines()
        if r.returncode or not lines:
            say(f'merge failed: {r.stderr[-500:]}'); return
        cand = int(lines[-1])
        if '--ids' in sys.argv and best_score is not None:
            cand = keep_outside_range(sys.argv[sys.argv.index('--ids') + 1])
        say(f'{changed} notebooks updated, {len(files)} files; candidate {cand}, best {best_score}')
        if best_score is None or cand < best_score:
            shutil.copy('out/candidate.csv', BEST); best_score = cand
    else:
        say(f'no notebook re-ran; best {best_score}, last submitted {last}')
    if last is not None and best_score >= last:
        say('no improvement; nothing submitted'); return
    if '--no-submit' in sys.argv:
        say('improved; --no-submit given'); return
    submit(best_score, 'auto refresh merge', say)


if __name__ == '__main__':
    main()
