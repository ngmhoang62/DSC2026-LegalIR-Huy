"""F4 non-CAL Jina+RRF pairwise specialist; CAL outcomes restricted to DEV."""
from __future__ import annotations
import gc, hashlib, json, math, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from benchmark_jina_reranker_holdouts import top_passages  # noqa:E402
from forensic_world_model import metrics,prepare_contract  # noqa:E402
from jina_ft_local import load_model_and_tokenizer,score_pairs  # noqa:E402
from run_burst_expanded_fusion_submission import DocumentStore  # noqa:E402
from tune_burst_phrases import multi_rrf  # noqa:E402
from tune_expanded_fusion_selection import ltr_features  # noqa:E402

OUT=ROOT/'results/sol_high_rl'; CACHE=ROOT/'cache/sol_high_rl'; DEV=('fold_0','fold_1','fold_2')
SCORES=CACHE/'noncal1050_jina_ft_boundary_scores.pkl'; MODEL=CACHE/'noncal_jina_boundary_specialist.pkl'

def paired(c,b,queries,ids):
    cv={q:len(set(c[q][:5])&queries[q][1])/len(queries[q][1]) for q in ids};bv={q:len(set(b[q][:5])&queries[q][1])/len(queries[q][1]) for q in ids}
    return {'wins':sum(cv[q]>bv[q] for q in ids),'losses':sum(cv[q]<bv[q] for q in ids),'ties':sum(cv[q]==bv[q] for q in ids),'changed_top5_sets':sum(set(c[q][:5])!=set(b[q][:5]) for q in ids)}

def candidate_features(ranking,scoremap,docs):
    pos={d:i+1 for i,d in enumerate(ranking)}; raw=np.asarray([scoremap[d] for d in docs],np.float64)
    logits=np.log(np.clip(raw,1e-5,1-1e-5)/np.clip(1-raw,1e-5,1)); z=(logits-logits.mean())/(logits.std() or 1.)
    out=[]
    for i,d in enumerate(docs):
        r=pos.get(d,1000);rr=[1/(k+r) for k in (0,2,5,10,20)]
        out.append(rr+[raw[i],z[i],z[i]*rr[0],z[i]*rr[3]])
    return np.asarray(out,np.float32)

def build_noncal_scores(train_ids,queries,cache):
    if SCORES.exists(): return pickle.loads(SCORES.read_bytes())
    docs=DocumentStore(sorted((ROOT/'DSC2026-LegalIR-main/v4_run/public_test_dataset/selected-contexts').glob('context_*.json')),cache_size=2048)
    model,tok=load_model_and_tokenizer(); result={}
    for qi,q in enumerate(train_ids,1):
        ranking=multi_rrf([[d for d,_ in source] for source in cache[q]],[.063,.357,.28,.30],5)
        selected=list(dict.fromkeys(ranking[:20]+[d for d in queries[q][1] if d in ranking]))
        owners=[];pairs=[]
        for d in selected:
            for passage in top_passages(queries[q][0],docs[d],count=2): owners.append(d);pairs.append((queries[q][0],passage))
        values=score_pairs(model,tok,pairs,batch_size=12,max_length=512); row={}
        for d,s in zip(owners,values):row[d]=max(row.get(d,-1e9),float(s))
        result[q]={'ranking':ranking,'scores':row,'selected':selected}
        if qi%50==0:
            SCORES.write_bytes(pickle.dumps(result,protocol=5));print(f'Jina boundary scores {qi}/{len(train_ids)}',flush=True)
    SCORES.write_bytes(pickle.dumps(result,protocol=5));del model,tok;gc.collect();torch.cuda.empty_cache();return result

def train_specialist(train_ids,queries,cache):
    if MODEL.exists(): return pickle.loads(MODEL.read_bytes())
    scored=build_noncal_scores(train_ids,queries,cache);xs=[];ys=[];weights=[];pairs=0
    for q in train_ids:
        row=scored[q]; ranking=row['ranking']; selected=row['selected']; idx={d:i for i,d in enumerate(selected)}
        x=candidate_features(ranking,row['scores'],selected);pos=[idx[d] for d in queries[q][1] if d in idx];neg=[idx[d] for d in ranking[3:20] if d not in queries[q][1] and d in idx]
        if not pos or not neg:continue
        w=1/(2*len(pos)*len(neg))
        for p in pos:
            for n in neg:
                diff=x[p]-x[n];xs.extend((diff,-diff));ys.extend((1,0));weights.extend((w,w));pairs+=1
    x=np.asarray(xs,np.float32);scaler=StandardScaler().fit(x);model=LogisticRegression(C=.1,solver='liblinear',max_iter=2000,random_state=2026)
    model.fit(scaler.transform(x),np.asarray(ys,np.int8),sample_weight=np.asarray(weights));acc=float(np.mean(model.predict(scaler.transform(x))==np.asarray(ys)))
    saved={'model':model,'scaler':scaler,'diagnostics':{'queries':len(train_ids),'unordered_pairs':pairs,'rows':len(ys),'in_sample_pair_accuracy_diagnostic_only':acc,'features':9,'C':.1}}
    MODEL.write_bytes(pickle.dumps(saved,protocol=5));return saved

def main():
    started=time.perf_counter();split_raw=(OUT/'CAL600_STRATIFIED_5FOLD_SEED42.json').read_bytes();split=json.loads(split_raw);br=json.loads((OUT/'CAL600_CANONICAL_BASELINE_REPORT.json').read_text(encoding='utf-8'))
    if hashlib.sha256(split_raw).hexdigest()!=br['protocol']['split_sha256']:raise RuntimeError('split mismatch')
    baseline=json.loads((OUT/'CAL600_CANONICAL_BASELINE_OOF_PREDICTIONS.json').read_text(encoding='utf-8'))
    queries,_,all_ids,candidates,views,scores,names,_,extras,_=prepare_contract();cbytes=(json.dumps({q:candidates[q] for q in all_ids},ensure_ascii=False,indent=2)+'\n').encode()
    if hashlib.sha256(cbytes).hexdigest()!=br['candidate_contract_sha256']:raise RuntimeError('candidate mismatch')
    raw=json.loads((ROOT/'DSC2026-LegalIR-main/v4_run/public_test_dataset/train.json').read_text(encoding='utf-8'));allq={str(q):(x['question'],{str(d) for d in x['answer']}) for q,x in raw.items() if x.get('answer')}
    feasible=json.loads((OUT/'BOUNDARY_SPECIALIST_FEASIBILITY.json').read_text(encoding='utf-8'));train_ids=[r['qid'] for r in feasible['rows']]
    retrieval=pickle.loads((ROOT/'results/burst_large_ltr/retrieval_train1000_tune50_val100.pkl').read_bytes())['cache'];saved=train_specialist(train_ids,allq,retrieval);model=saved['model'];scaler=saved['scaler']
    specialist={}
    for q in all_ids:
        ranking=views['base'][q];docs=candidates[q];x=candidate_features(ranking,scores['jina_ft'][q],docs);v=model.decision_function(scaler.transform(x));specialist[q]={d:float(s) for d,s in zip(docs,v)}
    nv=dict(views);ns=dict(scores);nn=list(names)+['external_jina_boundary'];ns['external_jina_boundary']=specialist;nv['external_jina_boundary']={q:sorted(candidates[q],key=lambda d:(-specialist[q][d],d)) for q in all_ids}
    rows,groups=ltr_features(nv,nn,candidates,all_ids,ns)
    for extra in extras:
        for q in all_ids:rows[q]=np.concatenate([rows[q],extra[q]],axis=1)
    pred={}
    for fold in DEV:
        test=split['folds'][fold];held=set(test);train=[q for q in all_ids if q not in held];x=np.vstack([rows[q] for q in train]);y=np.concatenate([[d in queries[q][1] for d in groups[q]] for q in train]).astype(np.int8);ss=StandardScaler().fit(x);m=LogisticRegression(C=.15,class_weight='balanced',solver='liblinear',max_iter=3000,random_state=2026).fit(ss.transform(x),y)
        for q in test:
            v=m.predict_proba(ss.transform(rows[q]))[:,1];pred[q]=[groups[q][i] for i in np.argsort(-v,kind='stable')]
    ids=sum((split['folds'][f] for f in DEV),[]);bm=metrics(baseline,queries,ids)[0];cm=metrics(pred,queries,ids)[0];multi=[q for q in ids if len(queries[q][1])>1];mb=metrics(baseline,queries,multi)[0];mc=metrics(pred,queries,multi)[0];folds={}
    for f in DEV:
        fi=split['folds'][f];b=metrics(baseline,queries,fi)[0];c=metrics(pred,queries,fi)[0];folds[f]={'delta':c['recall_at_5']-b['recall_at_5'],'baseline':b,'candidate':c,'paired':paired(pred,baseline,queries,fi)}
    delta=cm['recall_at_5']-bm['recall_at_5'];pp=paired(pred,baseline,queries,ids);md=mc['recall_at_5']-mb['recall_at_5'];promote=delta>=.005 and sum(x['delta']>0 for x in folds.values())>=2 and pp['wins']>pp['losses'] and md>=-.02
    standalone={q:sorted(candidates[q],key=lambda d:(-specialist[q][d],d)) for q in ids};report={'status':'PROMOTE_TO_CONFIRMATION' if promote else 'REJECT','family':'F4_external_jina_boundary','protocol':{'development_folds':list(DEV),'confirmation_outcomes_inspected':False,'noncal_overlap':0},'training':saved['diagnostics'],'standalone_dev':metrics(standalone,queries,ids)[0],'baseline_dev':bm,'candidate_dev':cm,'dev_delta':delta,'paired':pp,'folds':folds,'multi_gold':{'baseline':mb,'candidate':mc,'delta':md},'promotion_gate_passed':promote,'runtime_seconds':time.perf_counter()-started}
    (OUT/'F4_EXTERNAL_JINA_BOUNDARY_DEV_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(OUT/'F4_EXTERNAL_JINA_BOUNDARY_DEV_PREDICTIONS.json').write_text(json.dumps(pred,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'training':report['training'],'standalone':report['standalone_dev'],'delta':delta,'paired':pp,'fold_deltas':{f:x['delta'] for f,x in folds.items()},'multi_delta':md,'runtime_seconds':report['runtime_seconds']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
