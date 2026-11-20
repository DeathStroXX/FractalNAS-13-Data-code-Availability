"""Run-level statistics: per-method summaries, pooled LLM-vs-GA permutation tests, effect sizes,
Brown-Forsythe dispersion tests, equal-budget (first K unique architectures) analysis,
proxy/predictor fidelity. Writes results/stats.json."""
import itertools, json, os
import numpy as np
from scipy import stats
from common import *

def perm_p(a, b):
    """Exact two-sided permutation test on the difference of means."""
    allv = np.array(list(a) + list(b)); n = len(a); obs = np.mean(a) - np.mean(b); cnt = tot = 0
    for idx in itertools.combinations(range(len(allv)), n):
        m = np.zeros(len(allv), bool); m[list(idx)] = True
        tot += 1; cnt += abs(allv[m].mean() - allv[~m].mean()) >= abs(obs) - 1e-12
    return cnt / tot

def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9))

def ci95(a):
    a = np.asarray(a, float); return float(a.mean()), float(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))

def welch(a, b):
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)

def holm(pdict):
    items = sorted(pdict.items(), key=lambda kv: kv[1]); m = len(items); out = {}; prev = 0
    for i, (k, p) in enumerate(items):
        prev = max(prev, min(1.0, (m - i) * p)); out[k] = prev
    return out

def metrics(K):
    return {
        'best1': lambda x: x['acc1'].max(), 'avg1': lambda x: x['acc1'].mean(),
        'best3': lambda x: x['acc3'].max(), 'avg3': lambda x: x['acc3'].mean(),
        'top10_3': lambda x: np.sort(x['acc3'])[-10:].mean(),
        'nuniq': lambda x: len(x['acc1']), 'wall_h': lambda x: x['wall_h'],
        'best1@K': lambda x: x['acc1'][:K].max(), 'avg1@K': lambda x: x['acc1'][:K].mean(),
        'best3@K': lambda x: x['acc3'][:K].max(), 'avg3@K': lambda x: x['acc3'][:K].mean(),
        'top10_3@K': lambda x: np.sort(x['acc3'][:K])[-10:].mean(),
        'auc3@K': lambda x: np.maximum.accumulate(x['acc3'][:K]).mean(),
    }

if __name__ == '__main__':
    R = load_all(); S = {}
    for ds in DATASETS:
        K = int(min(len(x['acc1']) for m in METHODS for x in R[(ds, m)]))
        S[ds] = {'K': K, 'metrics': {}, 'runs': {}}
        for m in METHODS:
            S[ds]['runs'][m] = [dict(dir=x['dir'], n_raw=x['n_raw'], n_cached=x['n_cached'], nuniq=len(x['acc1']),
                                     best1=float(x['acc1'].max()), avg1=float(x['acc1'].mean()),
                                     best3=float(x['acc3'].max()), avg3=float(x['acc3'].mean()),
                                     bestP=float(x['pr'].max()), avgP=float(x['pr'].mean()), wall_h=float(x['wall_h']))
                                for x in R[(ds, m)]]
        for name, f in metrics(K).items():
            per = {m: [float(f(x)) for x in R[(ds, m)]] for m in METHODS}
            b = per['baseline']; l = [v for m in LLMS for v in per[m]]
            e = dict(per=per, ci={m: ci95(v) for m, v in per.items()}, ci_llm=ci95(l),
                     sd_ga=float(np.std(b, ddof=1)), sd_llm=float(np.std(l, ddof=1)),
                     perm_p=perm_p(l, b), g=float(hedges_g(l, b)),
                     bf_p=float(stats.levene(l, b, center='median').pvalue),
                     welch_per_model={m: welch(per[m], b) for m in LLMS},
                     separated=bool(min(l) > max(b)) or bool(max(l) < min(b)))
            S[ds]['metrics'][name] = e
        # Holm across the three per-model Welch tests x {best1, avg1} as in the original analysis
        S[ds]['welch_holm'] = holm({f'{m}|{k}': S[ds]['metrics'][k]['welch_per_model'][m] for m in LLMS for k in ['best1', 'avg1']})
        # proxy & predictor fidelity over all unique architectures
        A1 = np.concatenate([x['acc1'] for m in METHODS for x in R[(ds, m)]])
        A3 = np.concatenate([x['acc3'] for m in METHODS for x in R[(ds, m)]])
        P = np.concatenate([x['pr'] for m in METHODS for x in R[(ds, m)]])
        ov1, ovp, rank_of_best = [], [], []
        for m in METHODS:
            for x in R[(ds, m)]:
                t3 = set(np.argsort(-x['acc3'], kind='stable')[:10])
                ov1.append(len(t3 & set(np.argsort(-x['acc1'], kind='stable')[:10])) / 10)
                ovp.append(len(t3 & set(np.argsort(-x['pr'], kind='stable')[:10])) / 10)
                i = int(np.argmax(x['acc1'])); rank_of_best.append(int((x['acc3'] > x['acc3'][i]).sum() + 1))
        S[ds]['fidelity'] = dict(N=int(len(A1)),
            rho_1v3=float(stats.spearmanr(A1, A3)[0]), tau_1v3=float(stats.kendalltau(A1, A3)[0]),
            rho_pv3=float(stats.spearmanr(P, A3)[0]), tau_pv3=float(stats.kendalltau(P, A3)[0]),
            rho_pv1=float(stats.spearmanr(P, A1)[0]),
            mean_pred=float(P.mean()), mean_3=float(A3.mean()), mean_1=float(A1.mean()), max_pred=float(P.max()), max_3=float(A3.max()),
            n_distinct_pred=int(len(np.unique(P))), frac_pred_ge90=float((P >= 90).mean()),
            mean3_when_pred_ge90=float(A3[P >= 90].mean()) if (P >= 90).any() else None,
            top10_overlap_1ep=float(np.mean(ov1)), top10_overlap_pred=float(np.mean(ovp)),
            top10_overlap_random=float(10 / np.mean([len(x['acc1']) for m in METHODS for x in R[(ds, m)]])),
            rank3_of_1ep_best=rank_of_best)
    # primary endpoints, Holm-corrected across the four (dataset x {avg3, best3}) pooled tests
    for suffix in ['', '@K']:
        ph = holm({f'{ds}|{k}{suffix}': S[ds]['metrics'][k + suffix]['perm_p'] for ds in DATASETS for k in ['avg3', 'best3']})
        for key, v in ph.items():
            ds, k = key.split('|'); S[ds]['metrics'][k]['perm_p_holm'] = v
    # LoRA update statistics
    lora = {'examples': {}, 'seconds': []}
    for ds in DATASETS:
        for m in LLMS:
            for x in R[(ds, m)]:
                for e in x['evo']:
                    # an accepted rewrite earns a LoRA update only when the pipeline judged it
                    # novel or an improvement; the others carry no fine-tuning timestamps
                    if str(e.get('fine_tune_start_time')) in ('None', '') or not e.get('fine_tune_end_time') \
                            or str(e['fine_tune_end_time']) == 'None':
                        continue
                    k = str(e['train_examples_count']); lora['examples'][k] = lora['examples'].get(k, 0) + 1
                    from datetime import datetime as D
                    lora['seconds'].append((D.fromisoformat(e['fine_tune_end_time']) - D.fromisoformat(e['fine_tune_start_time'])).total_seconds())
    rew = [e['reward'] for ds in DATASETS for m in LLMS for x in R[(ds, m)] for e in x['evo']]
    S['lora'] = dict(examples=lora['examples'], median_s=float(np.median(lora['seconds'])), max_s=float(np.max(lora['seconds'])),
                     reward_cap=int(sum(r >= 24.99 for r in rew)), reward_lt1=int(sum(r < 1 for r in rew)), reward_negative=int(sum(r < 0 for r in rew)), n=len(rew),
                     n_finetuned=len(lora['seconds']))
    # every LLM-evolution record by state: accepted rewrite, rejected attempt, unchanged fallback
    S['evo_records'] = {}
    for ds in DATASETS:
        c = {}
        for m in LLMS:
            for r in raw_runs(m, ds):
                for e in r['evo_all']:
                    c[e['state']] = c.get(e['state'], 0) + 1
        S['evo_records'][ds] = c
    S['evo_records']['all'] = {k: sum(S['evo_records'][ds].get(k, 0) for ds in DATASETS)
                               for k in ('accepted_rewrite', 'rejected_attempt', 'unchanged_fallback')}
    S['evo_records']['all']['records'] = sum(S['evo_records']['all'].values())
    S['evo_records']['all']['segments'] = S['evo_records']['all']['accepted_rewrite'] + S['evo_records']['all']['unchanged_fallback']
    os.makedirs(RES, exist_ok=True)
    json.dump(S, open(os.path.join(RES, 'stats.json'), 'w'), indent=1)
    print('written', os.path.join(RES, 'stats.json'))
