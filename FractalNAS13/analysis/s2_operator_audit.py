"""Behavioural audit of every LLM-written operator recorded in the LLM-evolution logs.
1) extracts each accepted rewrite to analysis/operators/,
2) runs it in a sandbox for N randomized calls against a mock GA whose search space is Table I
   (gene order as listed in Table I), with a per-call timeout,
3) classifies it (priority order): raises/hangs > out-of-space > off-grid > behaviourally equivalent > sound modification
   using a 1% threshold of calls for the defect classes,
4) writes results/audit.json.
Behavioural equivalence is judged against the operators the pipeline actually ran
(genetic_algorithm_baseline.py): uniform initialisation, single-point crossover, k=3 tournament
selection, and a mutate_gene that redraws a gene with probability 2 x mutation_rate (0.4).
Note: operators are audited in isolation; how the framework handled failing calls at run time is not recorded in the logs."""
import os
from common import load_all, DATASETS, LLMS, RES, HERE
import glob,os,re,ast,textwrap,random,signal,math,json,collections
import numpy as np
from scipy import stats
SS={'n_columns':[2,3,4],'n_blocks':[2,3,4],'base_channels':[16,32,64],
 'dropout_prob':[0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5],'fc_dropout':[0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5],
 'lr':[0.01,0.008,0.006,0.005,0.004,0.003,0.002,0.001],'momentum':[0.75,0.8,0.85,0.88,0.9,0.92,0.95,0.98],
 'activation':['ReLU','GELU','LeakyReLU','SiLU'],'kernel_size':[1,3,5,7],'pooling_type':['Max','Avg'],
 'conv_type':['Standard','Depthwise'],'norm_type':['BatchNorm','InstanceNorm'],'optimizer_type':['SGD','AdamW','RMSprop']}
GENES=list(SS)
def valid(g,v):
    if isinstance(v,bool) or v is None: return False
    for x in SS[g]:
        if isinstance(x,str):
            if v==x: return True
        elif isinstance(v,(int,float)) and not isinstance(v,str) and math.isclose(v,x,rel_tol=1e-9,abs_tol=1e-12):
            if isinstance(x,int) and not float(v).is_integer(): continue
            return True
    return False
class TO(Exception): pass
def handler(s,f): raise TO()
signal.signal(signal.SIGALRM,handler)
class GA:
    def __init__(s):
        s.search_space={k:list(v) for k,v in SS.items()}; s.mutation_rate=0.2
        s.population=[]
        for i in range(20):
            c={g:random.choice(SS[g]) for g in GENES}; c['fitness']=random.uniform(10,80); s.population.append(c)
def load(code):
    code=textwrap.dedent(code)
    ns={'random':random,'math':math,'np':np,'copy':__import__('copy')}
    exec(compile(code,'<op>','exec'),ns)
    fn=[v for k,v in ns.items() if callable(v) and k in('mutate_gene','combine_genes','select_competitor','_create_random_chromosome')][0]
    return fn
def unreachable(code):
    t=ast.parse(textwrap.dedent(code)); n=0
    def term(st):
        if isinstance(st,(ast.Return,ast.Raise)): return True
        if isinstance(st,ast.If): return bool(st.orelse) and blk(st.body) and blk(st.orelse)
        return False
    def blk(b): return any(term(s) for s in b)
    for node in ast.walk(t):
        for fld in ('body','orelse','finalbody'):
            b=getattr(node,fld,None)
            if isinstance(b,list):
                for i,s in enumerate(b):
                    if term(s) and i<len(b)-1: n+=len(b)-1-i; break
    return n
def call(fn,*a,limit=0.2):
    signal.setitimer(signal.ITIMER_REAL,limit)
    try: return ('ok',fn(*a))
    except TO: return ('hang',None)
    except Exception as e: return ('exc',type(e).__name__)
    finally: signal.setitimer(signal.ITIMER_REAL,0)
def tourn_rank(pop,k=3):
    c=random.sample(pop,k); return max(c,key=lambda x:x['fitness'])
N=1500
def audit(fn,kind):
    random.seed(0); res=collections.Counter(); diffs=0; outs=collections.defaultdict(collections.Counter); hang_streak=0
    rk_base=collections.Counter(); rk=collections.Counter()
    for i in range(N):
        ga=GA()
        if kind=='mutate_gene':
            g=random.choice(GENES); cur=random.choice(SS[g])
            st,o=call(fn,ga,g,cur,list(SS[g]))
            if st=='ok':
                if not valid(g,o): st='invalid'
                elif o==cur: st='nochange'
                else: outs[g][o]+=1
        elif kind=='combine_genes':
            g=random.choice(GENES); p1=random.choice(SS[g]); p2=random.choice(SS[g]); cp=random.randint(1,12); gi=random.randint(0,12)
            st,o=call(fn,ga,g,p1,p2,cp,gi,13)
            if st=='ok':
                b=p1 if gi<cp else p2
                if not valid(g,o): st='invalid'
                elif o!=b: diffs+=1
        elif kind=='select_competitor':
            pop=sorted(ga.population,key=lambda x:-x['fitness'])
            st,o=call(fn,ga,ga.population)
            if st=='ok':
                try: rk[[id(x) for x in pop].index(id(o))]+=1
                except ValueError: st='invalid'
            rk_base[pop.index(tourn_rank(pop))]+=1
        else:
            st,o=call(fn,ga)
            if st=='ok':
                if not isinstance(o,dict) or set(o)!=set(GENES) or not all(valid(g,o[g]) for g in GENES): st='invalid'
                else:
                    for g in GENES: outs[g][o[g]]+=1
        res[st if st!='exc' else 'exc:'+str(o)]+=1
        if st=='hang': hang_streak+=1
        if hang_streak>=30: res['hang(extrapolated)']=N-i-1; break
    ok=sum(v for k,v in res.items() if k in('ok','nochange'))
    info={'calls':dict(res)}
    # behavioural equivalence
    eq=None
    if kind=='combine_genes': eq = (res.get('ok',0)==N and diffs==0); info['diff_from_single_point']=diffs/max(1,res.get('ok',0))
    if kind=='mutate_gene':
        unif=all(stats.chisquare([outs[g][v] for v in SS[g]] if False else list(outs[g].values())).pvalue>0.001 for g in outs if len(outs[g])>1)
        eq=(res.get('ok',0)==N and unif)
    if kind=='select_competitor':
        a=np.array([rk[i] for i in range(20)],float); b=np.array([rk_base[i] for i in range(20)],float)
        tv=0.5*np.abs(a/max(a.sum(),1)-b/b.sum()).sum() if a.sum()>0 else 1.0
        info['tv_vs_k3']=tv; info['mean_rank']=float((a*np.arange(1,21)).sum()/max(a.sum(),1)); info['mean_rank_base']=float((b*np.arange(1,21)).sum()/b.sum())
        eq=(res.get('ok',0)==N and tv<0.05)
    if kind=='_create_random_chromosome':
        unif=all(stats.chisquare([outs[g][v] for v in SS[g]]).pvalue>0.001 for g in GENES) if res.get('ok',0)==N else False
        eq=(res.get('ok',0)==N and unif)
    info['equivalent']=bool(eq); return info

def vclass(g,v):
    if valid(g,v): return 'ok'
    vals=SS[g]
    if isinstance(vals[0],str) or isinstance(v,(str,bool,list,dict)) or v is None: return 'oos'
    try:
        if min(vals)-1e-12<=v<=max(vals)+1e-12: return 'offgrid'
    except Exception: pass
    return 'oos'
N=1500
def audit2(fn,kind):
    random.seed(0); c=collections.Counter(); diffs=0; nok=0; hs=0
    outs=collections.defaultdict(collections.Counter); rk=collections.Counter(); rkb=collections.Counter()
    i=0
    while i<N:
        ga=GA(); i+=1
        if kind=='mutate_gene':
            # the pipeline's _mutate passes the gene's other values, never the current one
            g=random.choice(GENES); cur=random.choice(SS[g]); st,o=call(fn,ga,g,cur,[v for v in SS[g] if v!=cur])
            if st=='ok':
                st=vclass(g,o)
                if st=='ok':
                    if o==cur: st='nochange'
                    else: outs[g][o]+=1
        elif kind=='combine_genes':
            g=random.choice(GENES); p1=random.choice(SS[g]); p2=random.choice(SS[g]); cp=random.randint(1,12); gi=random.randint(0,12)
            st,o=call(fn,ga,g,p1,p2,cp,gi,13)
            if st=='ok':
                st=vclass(g,o)
                if st=='ok' and o!=(p1 if gi<cp else p2): diffs+=1
        elif kind=='select_competitor':
            pop=sorted(ga.population,key=lambda x:-x['fitness']); st,o=call(fn,ga,ga.population)
            if st=='ok':
                ids=[id(x) for x in pop]
                if id(o) in ids: rk[ids.index(id(o))]+=1
                else: st='oos'
            rkb[pop.index(tourn_rank(pop))]+=1
        else:
            st,o=call(fn,ga)
            if st=='ok':
                if not isinstance(o,dict) or set(o)!=set(GENES): st='oos'
                else:
                    cl=[vclass(g,o[g]) for g in GENES]
                    st='oos' if 'oos' in cl else ('offgrid' if 'offgrid' in cl else 'ok')
                    if st=='ok':
                        for g in GENES: outs[g][o[g]]+=1
        if st=='exc': st='exc'
        c[st]+=1
        hs = hs+1 if st=='hang' else 0
        if hs>=30: c['hang']+=N-i; break
    tot=sum(c.values()); fr={k:v/tot for k,v in c.items()}
    good=c['ok']+c['nochange']
    eq=False
    if fr.get('ok',0)+fr.get('nochange',0)>0.999:
        if kind=='combine_genes': eq= diffs==0
        # the operator the rewrite replaced is the pipeline's own mutate_gene: it redraws the gene
        # with probability 2 x mutation_rate = 0.4, uniformly over the other values, and otherwise
        # leaves it unchanged
        elif kind=='mutate_gene':
            p_keep = 1 - min(1.0, 2 * GA().mutation_rate)
            eq = (stats.binomtest(c['nochange'], good, p_keep).pvalue > 0.001
                  and all(stats.chisquare(list(outs[g].values())+[0]*(len(SS[g])-len(outs[g]))).pvalue>0.001 for g in outs))
        elif kind=='select_competitor':
            a=[rk[j] for j in range(20)]; b=[rkb[j] for j in range(20)]
            tab=np.array([a,b]); tab=tab[:,tab.sum(0)>0]; eq= stats.chi2_contingency(tab)[1]>0.01
        else: eq= all(stats.chisquare([outs[g][v] for v in SS[g]]).pvalue>0.001 for g in GENES)
    mean_rank=sum((j+1)*rk[j] for j in rk)/max(1,sum(rk.values())); base_rank=sum((j+1)*rkb[j] for j in rkb)/max(1,sum(rkb.values()))
    return fr,eq,diffs/max(1,good),mean_rank,base_rank

if __name__=='__main__':
    R=load_all(); rows=[]; opdir=os.path.join(HERE,'operators'); os.makedirs(opdir,exist_ok=True)
    for ds in DATASETS:
        for mod in LLMS:
            for run,x in enumerate(R[(ds,mod)],1):
                for e in x['evo']:
                    kind=e['method']; body=e['cleaned_code'] or ''
                    fn_name=f"{ds}_{mod}_r{run}_it{e['attempt']}_{kind}.py"
                    open(os.path.join(opdir,fn_name),'w').write(f"# {ds} {mod} run{run} meta-iteration {e['attempt']} | inner-GA best 3-epoch acc {e['score']} | reward {e['reward']:.2f}\n"+body+"\n")
                    fn=load(body); fr,eq,dif,mr,br=audit2(fn,kind)
                    fail=fr.get('exc',0)+fr.get('hang',0); oos=fr.get('oos',0); off=fr.get('offgrid',0)
                    if fail>=0.01: cat='raises/hangs'
                    elif oos>=0.01: cat='out-of-space'
                    elif off>=0.01: cat='off-grid'
                    elif eq: cat='equivalent'
                    else: cat='modified'
                    rows.append(dict(ds=ds,model=mod,run=run,it=int(e['attempt']),op=kind,file=fn_name,fail=float(fail),oos=float(oos),off=float(off),
                        equivalent=bool(eq),diff_from_single_point=float(dif),mean_selected_rank=float(mr),baseline_rank=float(br),
                        cat=cat,unreachable_stmts=int(unreachable(body)),code_lines=len([l for l in body.splitlines() if l.strip() and not l.strip().startswith('#')]),
                        inner_best3=float(e['score']),reward=float(e['reward'])))
                    print(fn_name,cat)
    json.dump(rows,open(os.path.join(RES,'audit.json'),'w'),indent=1)
