"""Shared loading utilities. All paths are relative to the project root (one level up).

Arms
----
baseline : the handcrafted GA, one long run per replicate (caches duplicate evaluations)
frozen   : the restart schedule with handcrafted, *frozen* operators and no LLM in the loop
           (the schedule control; 5 replicates per dataset)
qwen / deepseek / mistral : the restart schedule with LLM-rewritten operators (3 replicates each)

`frozen` and the three LLM arms share the same schedule (5 inner GA runs of 5 generations,
population 20, fresh random population per inner run) and neither of them caches, so
frozen-vs-LLM isolates the operator code and frozen-vs-baseline isolates the schedule.
"""
import json, glob, os, re, sys
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------- determinism
# The 90 extracted operators are ordinary Python written by a language model, and several of
# them iterate over a set of gene names. Python randomises string hashing per process, so the
# iteration order of such a set -- and therefore the operator's behaviour, and therefore the
# replayed effect of that operator -- changes from one interpreter launch to the next. Pinning
# the hash seed is what makes `s20_operator_bench.py` and everything downstream of it
# reproduce exactly rather than approximately. The variable has to be set before the
# interpreter starts, so the first import re-executes the script once with it set.
if os.environ.get('PYTHONHASHSEED') != '0' and os.path.isfile(sys.argv[0] or ''):
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('NAS_LOGS', os.path.join(HERE, '..', 'logs'))
#: The 34 pipeline pod logs (~330 MB) are archived separately, in fractalnas13_large_input_data.zip; unzip it
#: next to analysis/. Same <arm>/<dataset>/<run> layout as logs/; only the scripts that parse
#: generations, segments or throughput need them (s14, s28, segments()).
POD_ROOT = os.environ.get('NAS_POD_LOGS', os.path.join(HERE, '..', 'large_input_data', 'pod_logs'))
#: The text an LLM-evolution record carries when every attempt at an iteration failed and the
#: segment ran on the previous working code instead.
FALLBACK = 'Fallback triggered. Code unchanged.'
RES = os.path.join(HERE, 'results')
DATA = os.path.join(HERE, '..', 'data')

METHODS = ['baseline', 'frozen', 'qwen', 'deepseek', 'mistral']
LLMS = ['qwen', 'deepseek', 'mistral']
RESTART = ['frozen'] + LLMS           # every arm that uses the restart schedule
DATASETS = ['cifar10', 'cifar100']
LABEL = {'baseline': 'Baseline GA', 'frozen': 'Frozen operators (schedule control)',
         'qwen': 'Qwen2.5-Coder-7B', 'deepseek': 'DeepSeek-Coder-6.7B', 'mistral': 'Mistral-7B'}
SHORT = {'baseline': 'Baseline GA', 'frozen': 'Restart, frozen ops',
         'qwen': 'Qwen2.5-Coder-7B', 'deepseek': 'DeepSeek-Coder-6.7B', 'mistral': 'Mistral-7B'}

GENES = ['n_columns', 'base_channels', 'n_blocks', 'dropout_prob', 'fc_dropout', 'lr', 'momentum',
         'activation', 'kernel_size', 'pooling_type', 'conv_type', 'norm_type', 'optimizer_type']


def _rd(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def ts(x):
    return datetime.fromisoformat(x['timestamp'])


def _run_dirs(method, ds):
    """Every replicate directory of an arm: any subdirectory holding a *_true.jsonl log.

    The arms use different directory-naming conventions (`25-09-26_15-23` for the
    baseline and the LLM arms, `2026-10-05_03-48-22_run1_seed_2001` for the frozen
    control), so the presence of the logs decides, not the name.
    """
    out = []
    for d in sorted(glob.glob(f'{ROOT}/{method}/{ds}/*')):
        if os.path.isdir(d) and glob.glob(d + '/*_true.jsonl'):
            out.append(d)
    return out


def evo_state(e):
    """Classify one LLM-evolution record.

    The pipeline logs every attempt. An iteration makes up to ten attempts; the first one whose
    code parses and passes the smoke call is spliced in (`accepted_rewrite`), each failed one is
    logged and reverted (`rejected_attempt`), and if all ten fail the segment runs on the previous
    working code and is closed by a record whose code is the FALLBACK marker
    (`unchanged_fallback`). Exactly one accepted rewrite or fallback closes each of a run's five
    segments.
    """
    code = (e.get('cleaned_code') or '').strip()
    if code == FALLBACK:
        return 'unchanged_fallback'
    return 'accepted_rewrite' if _truthy(e.get('valid_syntax')) else 'rejected_attempt'


def _truthy(v):
    return v is True or str(v).strip().lower() == 'true'


def raw_runs(method, ds):
    """Per run: e1 = 1-epoch log (*_1_epoch.jsonl), e3 = 3-epoch log (*_true.jsonl),
    pr = LLM accuracy-predictor log (no suffix), meta = frozen-control meta log (frozen arm only).

    LLM arms: `evo_all` is every record of the LLM-evolution log (accepted rewrites, rejected
    attempts and unchanged fallbacks, in log order); `evo` is the accepted rewrites only -- the
    operators the language model wrote that the pipeline spliced into the search, which is what
    the operator analyses audit and replay; `evo_close` is the record that closed each of the five
    segments (an accepted rewrite or an unchanged fallback)."""
    out = []
    for d in _run_dirs(method, ds):
        fs = glob.glob(d + '/*.jsonl')
        ev = [f for f in fs if 'LLM-evolution' in f]
        ml = [f for f in fs if 'meta_log' in f]
        ga = [f for f in fs if 'LLM-evolution' not in f and 'meta_log' not in f]
        e1 = [f for f in ga if f.endswith('_1_epoch.jsonl')][0]
        e3 = [f for f in ga if f.endswith('_true.jsonl')][0]
        pr = [f for f in ga if not f.endswith('_1_epoch.jsonl') and not f.endswith('_true.jsonl')][0]
        evo_all = _rd(ev[0]) if ev else None
        for e in evo_all or []:
            e['state'] = evo_state(e)
        out.append(dict(dir=os.path.basename(d), path=d, e1=_rd(e1), e3=_rd(e3), pr=_rd(pr),
                        evo_all=evo_all,
                        evo=[e for e in evo_all if e['state'] == 'accepted_rewrite'] if ev else None,
                        evo_close=[e for e in evo_all if e['state'] != 'rejected_attempt'] if ev else None,
                        meta=_rd(ml[0]) if ml else None))
    return out


def runs(method, ds):
    """Unique architectures in order of first evaluation, aligned across the three logs."""
    res = []
    for r in raw_runs(method, ds):
        seen, u1 = set(), []
        for x in sorted(r['e1'], key=ts):
            if x['uid'] in seen:
                continue
            seen.add(x['uid']); u1.append(x)
        d3 = {x['uid']: x['accuracy'] for x in r['e3']}
        dp = {x['uid']: x['accuracy'] for x in r['pr']}
        t = [ts(x) for x in r['e1']]
        res.append(dict(dir=r['dir'], method=method, dataset=ds,
                        uids=[x['uid'] for x in u1],
                        acc1=np.array([x['accuracy'] for x in u1]),
                        acc3=np.array([d3[x['uid']] for x in u1]),
                        pr=np.array([dp[x['uid']] for x in u1]),
                        it=np.array([x.get('meta_iteration') or 0 for x in u1]),
                        chrom=[x.get('chromosome') for x in u1],
                        n_raw=len(r['e1']),
                        n_cached=sum(bool(x.get('is_cached', False)) for x in r['e1']),
                        wall_h=(max(t) - min(t)).total_seconds() / 3600,
                        evo=r['evo'], meta=r['meta']))
    return res


def load_all():
    return {(ds, m): runs(m, ds) for ds in DATASETS for m in METHODS}


def segments(method, ds):
    """Split every run into its inner search segments.

    Restart arms with an LLM: one segment per meta-iteration (the inner 5-generation GA run),
    delimited by the LoRA fine-tuning timestamps in the LLM-evolution log (validated: the best
    3-epoch accuracy of every segment equals the inner-run score logged for that iteration, 90/90).
    Frozen control: one segment per `meta_iteration` field, which the frozen logs record directly.
    Baseline runs: consecutive windows of 100 logged evaluations (= 5 generations of 20, cache hits
    included), so that a baseline window spans the same nominal search effort as one inner run.
    """
    out = []
    for r in raw_runs(method, ds):
        e1 = sorted(r['e1'], key=ts)
        d1 = {x['uid']: x['accuracy'] for x in r['e1']}
        d3 = {x['uid']: x['accuracy'] for x in r['e3']}
        segs = []
        if r['evo'] is not None:
            # A segment is the inner GA run under one `=== Iteration i/5` banner of the pod log.
            # Each evaluation is assigned to its banner by aligning the JSONL records, in time
            # order, with the pod log's `Evaluating unique arch (checksum: ...)` lines (the log is
            # a superset: it also prints architectures the accuracy floor later removed). Not
            # every closing record carries fine-tuning timestamps -- fallbacks and rewrites that
            # earned no fine-tune have none -- so the banners, not the timestamps, delimit.
            # An attempt that fails after its GA has started (`[Meta] ✗ Iteration i (Attempt ...)
            # failed`) is reverted; the evaluations it made belong to no segment (label None).
            events = [('it', int(m.group(1))) if m.group(1) else ('fail', None) if m.group(2)
                      else ('arch', m.group(3))
                      for m in re.finditer(r'=== Iteration (\d+)/\d+|(\[Meta\] \S+ Iteration \d+ \(Attempt \d+/\d+\) failed)'
                                           r'|Evaluating unique arch \(checksum:\s*([0-9a-f]+)',
                                           _pod_text(r['path']))]
            it_of, cur, j, start = [], None, 0, 0
            for x in e1:
                while j < len(events):
                    kind, v = events[j]; j += 1
                    if kind == 'it':
                        cur, start = v, len(it_of)
                    elif kind == 'fail':
                        it_of[start:] = [None] * (len(it_of) - start)
                        start = len(it_of)
                    elif x['uid'].startswith(v):
                        break
                else:
                    raise ValueError(f'pod log does not cover {method}/{ds}/{r["dir"]}')
                it_of.append(cur)
            if any(k == 'fail' for k, _ in events[j:]):   # a failure after the last aligned record
                tail = [k for k, _ in events[j:]]
                if 'it' not in tail[:tail.index('fail')]:
                    it_of[start:] = [None] * (len(it_of) - start)
            close = r['evo_close']
            for k, e in enumerate(close, 1):
                seg = [x for x, i in zip(e1, it_of) if i == k]
                segs.append(dict(it=k, method=e['method'], raw=seg, evo=e, state=e['state']))
        elif any(x.get('meta_iteration') for x in e1):
            by = {}
            for x in e1:
                by.setdefault(x['meta_iteration'], []).append(x)
            for k in sorted(by):
                segs.append(dict(it=int(k), method='frozen_identity', raw=by[k], evo=None))
        else:
            for w in range(len(e1) // 100):
                segs.append(dict(it=w + 1, method=None, raw=e1[w * 100:(w + 1) * 100], evo=None))
        for s in segs:
            u = list(dict.fromkeys(x['uid'] for x in s['raw']))
            s['uids'] = u
            s['acc1'] = np.array([d1[k] for k in u])
            s['acc3'] = np.array([d3[k] for k in u])
            s['n_raw'] = len(s['raw'])
            s['cache_share'] = float(np.mean([bool(x.get('is_cached', False)) for x in s['raw']]))
            s['t0'] = ts(s['raw'][0]); s['t1'] = ts(s['raw'][-1])
        out.append(dict(dir=r['dir'], method=method, dataset=ds, segs=segs, t0=min(ts(x) for x in e1)))
    return out


# ------------------------------------------------------------------ generations
_GEN = re.compile(r'Generation:\s*(\d+)\s*/\s*(\d+)')
_ARCH = re.compile(r'Evaluating unique arch \(checksum:\s*([0-9a-f]+)')


_SHARED = {}


def _pod_files(d):
    """The pod-log files that belong to THIS run.

    One multi-run pod log (`cifar100_frozen_Pod_run1_to_run4_logs`) was archived as an identical
    copy inside four run directories. Counting it once per directory would quadruple everything it
    contains, so any file whose contents also appear in a sibling run directory is treated as a
    shared log and skipped; each of those runs also has its own `pod_logs_N.txt`.
    """
    import hashlib
    rel = os.path.relpath(os.path.abspath(d), os.path.abspath(ROOT))
    moved = os.path.join(POD_ROOT, rel)
    if not rel.startswith('..') and os.path.isdir(moved):
        d = moved
    parent = os.path.dirname(os.path.abspath(d))
    if parent not in _SHARED:
        seen, shared = {}, set()
        for sib in sorted(glob.glob(os.path.join(parent, '*'))):
            if not os.path.isdir(sib):
                continue
            for pat in ('*.log', '*.txt', '*logs'):
                for f in sorted(glob.glob(os.path.join(sib, pat))):
                    try:
                        h = hashlib.md5(open(f, 'rb').read()).hexdigest()
                    except OSError:
                        continue
                    if h in seen and seen[h] != os.path.dirname(f):
                        shared.add(h)
                    seen.setdefault(h, os.path.dirname(f))
        _SHARED[parent] = shared
    out = []
    for pat in ('*.log', '*.txt', '*logs'):
        for f in sorted(glob.glob(os.path.join(d, pat))):
            try:
                h = hashlib.md5(open(f, 'rb').read()).hexdigest()
            except OSError:
                continue
            if h not in _SHARED[parent]:
                out.append(f)
    return out


def _pod_text(d):
    txt = []
    for f in _pod_files(d):
        try:
            txt.append(open(f, errors='ignore').read())
        except OSError:
            pass
    return '\n'.join(txt)


def generations(method, ds):
    """Map every evaluated architecture to the generation that produced it, from the pod log.

    The pipeline prints `Generation: g/N` before each individual and
    `Evaluating unique arch (checksum: ...)` for each architecture it actually trains, so the two
    interleaved streams give the true generation label. The baseline runs one long GA of 26
    generations; each segment of a restart run runs 5, and a reset of g to 1 starts a new segment.

    Returns, per run: {'dir', 'n_gen', 'label': {uid -> (segment, generation)}}. Architectures the
    log mentions but the released JSONL does not contain (removed by the pipeline's accuracy floor)
    are skipped, as are checksum prefixes that match more than one uid.
    """
    out = []
    for r in raw_runs(method, ds):
        uids = {x['uid'] for x in r['e1']}
        by_prefix = {}
        for u in uids:
            by_prefix.setdefault(u[:8], []).append(u)
        txt = _pod_text(r['path'])
        label, seg, gen, prev = {}, 0, None, None
        n_gen = 0
        for m in re.finditer(r'Generation:\s*(\d+)\s*/\s*(\d+)|Evaluating unique arch \(checksum:\s*([0-9a-f]+)', txt):
            if m.group(1):
                g, tot = int(m.group(1)), int(m.group(2))
                n_gen = max(n_gen, tot)
                if prev is not None and g < prev:
                    seg += 1
                if prev is None:
                    seg = 1
                gen, prev = g, g
            else:
                cand = by_prefix.get(m.group(3), [])
                if gen is not None and len(cand) == 1:
                    label.setdefault(cand[0], (seg, gen))
        out.append(dict(dir=r['dir'], method=method, dataset=ds, n_gen=n_gen, label=label,
                        n_logged=len(uids), n_labelled=len(label)))
    return out


# ------------------------------------------------------------------ chromosomes
def gene_table():
    """One row per architecture a run evaluated, with its 13 genes, for every arm.

    The baseline and LLM arms take their chromosomes from the released per-architecture
    record (data/gene_diversity_benchmark_updated.csv); the frozen control takes them from
    its own logs, which carry the chromosome inline.

    The released record has no run column. It lists, for each dataset and arm, the runs one
    after another, each as its unique architectures in order of first evaluation -- the same
    order `runs()` reads out of the logs. A row is therefore attached to its run by position
    and checked against that run's `uid`. Attaching by `uid` alone is wrong: runs that share
    a seed share their initial population, so one `uid` is evaluated by several runs (each
    with its own measured accuracy), and a lookup by `uid` hands all of them to one run.
    """
    import csv
    CSVMAP = {'LR': 'lr', 'Momentum': 'momentum', 'Activation': 'activation', 'Kernel': 'kernel_size',
              'Pooling': 'pooling_type', 'Conv_Type': 'conv_type', 'Norm_Type': 'norm_type',
              'Optimizer': 'optimizer_type', 'FC_Dropout': 'fc_dropout', 'Dropout_Prob': 'dropout_prob',
              'N_Columns': 'n_columns', 'Base_Channels': 'base_channels', 'N_Blocks': 'n_blocks'}
    # (dataset, arm) -> the (run id, uid) of every row that arm contributes, in record order
    expect = {}
    for m in ['baseline'] + LLMS:
        for ds in DATASETS:
            expect[(ds, m)] = [(f"{m}/{ds}/{r['dir']}", u) for r in runs(m, ds) for u in r['uids']]
    cursor = {k: 0 for k in expect}
    rows = []
    path = os.path.join(DATA, 'gene_diversity_benchmark_updated.csv')
    with open(path, newline='') as fh:
        for rec in csv.DictReader(fh):
            ds = rec['dataset']
            key = (ds, rec['pipeline'])
            if key not in expect or cursor[key] >= len(expect[key]):
                raise ValueError('record row %s / %s has no run left to belong to' % key)
            rid, uid = expect[key][cursor[key]]
            cursor[key] += 1
            if uid != rec['uid']:
                raise ValueError('record row %s of %s / %s is %s, but the logs have %s there'
                                 % (cursor[key], ds, rec['pipeline'], rec['uid'], uid))
            row = {g: rec[k] for k, g in CSVMAP.items()}
            row.update(uid=rec['uid'], dataset=ds, method=rec['pipeline'],
                       arm='baseline' if rec['pipeline'] == 'baseline' else 'llm',
                       run=rid, acc1=_num(rec['acc_epoch1']), acc3=_num(rec['acc_epoch3_true']),
                       pred=_num(rec['predicted_fitness']))
            rows.append(row)
    short = {k: len(v) - cursor[k] for k, v in expect.items() if cursor[k] != len(v)}
    if short:
        raise ValueError('the record is missing rows the logs have: %s' % short)
    for ds in DATASETS:
        for r in runs('frozen', ds):
            for u, c, a1, a3, p in zip(r['uids'], r['chrom'], r['acc1'], r['acc3'], r['pr']):
                if c is None:
                    continue
                row = {g: c.get(g) for g in GENES}
                row.update(uid=u, dataset=ds, method='frozen', arm='frozen',
                           run=f"frozen/{ds}/{r['dir']}", acc1=float(a1), acc3=float(a3), pred=float(p))
                rows.append(row)
    return rows


def _num(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None
