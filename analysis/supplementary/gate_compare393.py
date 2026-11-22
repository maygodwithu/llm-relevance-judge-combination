import gzip,json,glob,os,numpy as np
R='cpmr_experiment'
def m(path):
    raw=json.load(gzip.open(path,'rt'))['raw']
    return {(q,d):v[0]+v[1]-v[3]-v[4] for q in raw for d,v in raw[q].items()}
pairs=[('Base__T2__gate','esciLlamaEval_Base__T2__sel.json.gz'),('ELQ_mix_T2_s0_step625__T2__gate','esci_204/eval/esciLlamaEval_ELQ_mix_T2_s0_step625__T2__sel.json.gz')]
out=[]
for g,s in pairs:
    gf=R+f'/esci393/gate/esciLlamaEval_{g}.json.gz'
    sf=[f for f in glob.glob(R+'/**/'+os.path.basename(s),recursive=True) if 'llama' in str(json.load(gzip.open(f,'rt'))['meta']['model_name']).lower() or True]
    sf=glob.glob(R+'/'+s) or [f for f in glob.glob(R+'/**/'+os.path.basename(s),recursive=True)]
    a=m(gf); b=m(sf[0]); keys=list(a); d=np.array([abs(a[k]-b[k]) for k in keys])
    out.append(dict(job=g,n=len(keys),max_abs_diff=float(d.max()),mean_abs_diff=float(d.mean()),identical=int((d==0).sum()),decision_disagreements=int(sum((a[k]>0)!=(b[k]>0) for k in keys)),passed=bool(d.max()<=1e-4),criterion=1e-4))
json.dump(out,open(R+'/esci393/gate/gate393.json','w'),indent=1); print(out)
