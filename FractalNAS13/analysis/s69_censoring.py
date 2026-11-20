"""Left truncation of the record: the share of architecture-training attempts the pipeline's
accuracy floor removed before logging, per configuration (arm x dataset).

Every architecture the pipeline trains is announced in the pod log by an
`Evaluating unique arch (checksum: ...)` line; one that ends below the dataset's first-epoch
floor prints `Error during training: (..., "Accuracy is too low ...")` and is never written to
the JSONL logs. The share of the second among the first is the truncation rate the
supplement's threats section quotes. Reads large_input_data/pod_logs/.

Writes results/censoring.json.
"""
import json, os, re
import common

ARCH = re.compile(r'Evaluating unique arch \(checksum:')
FLOOR = re.compile(r'Error during training: \([^\n]*Accuracy is too low')


def main():
    out = {'by_config': {}, 'total': {'attempts': 0, 'floor': 0}}
    for ds in common.DATASETS:
        for m in common.METHODS:
            att = flo = 0
            for r in common.raw_runs(m, ds):
                txt = common._pod_text(r['path'])
                att += len(ARCH.findall(txt))
                flo += len(FLOOR.findall(txt))
            out['by_config'][f'{m}/{ds}'] = dict(attempts=att, floor=flo, rate=flo / att)
            out['total']['attempts'] += att
            out['total']['floor'] += flo
            print(f'{ds:9s} {m:9s} {flo:5d} of {att:5d} attempts  {100 * flo / att:5.1f}%')
    for ds in common.DATASETS:
        rates = [v['rate'] for k, v in out['by_config'].items() if k.endswith('/' + ds)]
        out[ds] = dict(min=min(rates), max=max(rates))
        print(f'{ds}: {100 * min(rates):.1f}--{100 * max(rates):.1f}% by configuration')
    print('total', out['total'])
    json.dump(out, open(os.path.join(common.RES, 'censoring.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
