"""Check that the archived figures, tables and statistics are reproduced exactly.

    python3 verify_results.py

Copies this folder to a temporary directory, runs `run_all.sh figures`, `tables` and `stats`
there, and compares every output with the archived copy: tables, results and PNG figures byte
for byte, PDF figures with their embedded timestamps ignored. Nothing in this folder is
modified. Exit status 0 when everything matches.
"""
import filecmp
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def same_pdf(a, b):
    strip = lambda p: re.sub(rb'/(CreationDate|ModDate) \([^)]*\)', b'', open(p, 'rb').read())
    return strip(a) == strip(b)


def main():
    tmp = tempfile.mkdtemp(prefix='fractalnas13_verify_')
    work = os.path.join(tmp, 'pkg')
    shutil.copytree(HERE, work, ignore=shutil.ignore_patterns('__pycache__'))
    env = dict(os.environ, PYTHON=os.environ.get('PYTHON', sys.executable))
    for target in ('figures', 'tables', 'stats'):
        r = subprocess.run(['bash', 'run_all.sh', target], cwd=work, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode:
            print(r.stdout[-3000:])
            print('FAILED: run_all.sh %s' % target)
            return 1
    diff = []
    for sub in ('tables', 'figures', os.path.join('analysis', 'results')):
        for f in sorted(os.listdir(os.path.join(HERE, sub))):
            a, b = os.path.join(HERE, sub, f), os.path.join(work, sub, f)
            ok = same_pdf(a, b) if f.endswith('.pdf') else filecmp.cmp(a, b, shallow=False)
            if not ok:
                diff.append(os.path.join(sub, f))
    shutil.rmtree(tmp)
    if diff:
        print('DIFFERENT from the archived copy (%d):' % len(diff))
        for d in diff:
            print('  ' + d)
        return 1
    print('every figure, table and result is identical to the archived copy')
    return 0


if __name__ == '__main__':
    sys.exit(main())
