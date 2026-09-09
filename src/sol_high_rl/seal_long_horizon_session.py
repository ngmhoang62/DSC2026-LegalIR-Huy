"""Seal long-horizon campaign state and immutable summary from experiment reports."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/sol_high_rl'
read=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
anatomy=read('RANKING_FAILURE_ANATOMY_V2.json');f1=read('F1_FIXED_CAPACITY_DEV_REPORT.json');f2=read('F2_EXPERT_FACILITY_CONFIRM_REPORT.json');f3=read('F3_NONCAL_BOUNDARY_METRIC_DEV_REPORT.json');f4=read('F4_EXTERNAL_JINA_BOUNDARY_DEV_REPORT.json');baseline=read('CAL600_CANONICAL_BASELINE_REPORT.json')
summary={
 'status':'CONVERGED_NO_SUBMISSION_CANDIDATE','session':'RL_LONG_HORIZON','population':'CAL600',
 'scientific_ground_truth':{'split_sha256':baseline['protocol']['split_sha256'],'candidate_contract_sha256':baseline['candidate_contract_sha256'],'baseline':baseline['baseline'],'candidate_ceiling':baseline['candidate_ceiling']['pooled'],'old_lobo_anchor':baseline['old_4block_lobo_anchor']['recall_at_5'],'public_external_anchor':.954750001},
 'failure_anatomy':{'artifact':'results/sol_high_rl/RANKING_FAILURE_ANATOMY_V2.json','candidate_oracle':anatomy['candidate_oracle'],'headroom':anatomy['headroom'],'miss_occurrences':anatomy['miss_decomposition']['occurrence_counts'],'failure_mass':anatomy['miss_decomposition']['exclusive_failure_class_recall_mass'],'multi_gold':{k:anatomy['multi_gold'][k] for k in ('queries','baseline_recall','candidate_oracle_recall','expert_top5_union_recall','queries_with_reachable_missed_gold')}},
 'anti_adaptive_protocol':{'development_folds':['fold_0','fold_1','fold_2'],'confirmation_folds':['fold_3','fold_4'],'confirmation_families_used':1,'confirmation_family_limit':2,'confirmation_outcomes_only_opened_for':['F2_expert_facility_slate']},
 'families':[
  {'id':'F1','mechanism':'fixed historical pairwise and shallow LambdaRank capacity','status':'REJECT_DEV','runtime_seconds':f1['runtime_seconds'],'best_dev_delta':f1['variants'][f1['predeclared_champion']]['dev_recall_delta'],'paired':f1['variants'][f1['predeclared_champion']]['paired'],'fold_deltas':{k:v['recall_delta'] for k,v in f1['variants'][f1['predeclared_champion']]['folds'].items()}},
  {'id':'F2','mechanism':'label-free submodular expert-facility slate','status':'REJECT_CONFIRM','dev_delta':read('F2_EXPERT_FACILITY_DEV_REPORT.json')['dev_delta'],'confirmation_delta':f2['confirmation']['delta'],'confirmation_paired':f2['confirmation']['paired'],'confirmation_fold_deltas':{k:v['delta'] for k,v in f2['confirmation']['folds'].items()},'runtime_seconds':f2['runtime_seconds']},
  {'id':'F3','mechanism':'non-CAL diagonal full-body boundary metric','status':'REJECT_BELOW_DEV_GATE_WEAK_POSITIVE','dev_delta':f3['dev_delta'],'paired':f3['paired'],'fold_deltas':{k:v['delta'] for k,v in f3['folds'].items()},'multi_delta':f3['multi_gold']['delta'],'runtime_seconds':f3['runtime_seconds']},
  {'id':'F4','mechanism':'non-CAL Jina plus RRF boundary calibration','status':'REJECT_DEV','dev_delta':f4['dev_delta'],'paired':f4['paired'],'fold_deltas':{k:v['delta'] for k,v in f4['folds'].items()},'multi_delta':f4['multi_gold']['delta'],'runtime_seconds':f4['runtime_seconds']},
 ],
 'closed_or_deferred':{'full_neural_boundary_finetune':'DEFER_LOW_EV: objective provenance unknown, consensus-blind mass 0.001667, F3 below gate and F4 negative','selective_aiteam_candidate_rescue':'NOT_ELIGIBLE: ranking did not approach candidate ceiling','router_dynamic_ensemble':'CLOSED_PREMISE: no robust complementary second system'},
 'submission':{'created':False,'uploaded':False,'reason':'no mechanism passed canonical confirmation and submission-level effect gate'},
 'next_highest_ev_branch':None,
 'uncertainty':'CAL600 remains small and repeatedly researched; F3 direction may be real but is below the preregistered material-effect threshold. New external training provenance or a demonstrably different evidence model would be required to reopen representation work.'
}
path=OUT/'SESSION_RL_LONG_HORIZON_SUMMARY.json';path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
world=read('WORLD_MODEL.json');world['status']=summary['status'];world['long_horizon_session']=summary;world['anti_adaptive_protocol']['confirmation_families_used']=1;world['current_beliefs']['next_single_hypothesis']=None;world['current_beliefs']['convergence_reason']='F1 capacity rejected; F2 reversed on confirmation; F3 below gate; F4 negative. No robust second system and no eligible selective-rescue premise.'
tmp=(OUT/'WORLD_MODEL.json.tmp');tmp.write_text(json.dumps(world,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(OUT/'WORLD_MODEL.json')
print(json.dumps({'summary':str(path),'summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'world_status':world['status'],'families':[(x['id'],x['status']) for x in summary['families']]},ensure_ascii=False,indent=2))
if __name__=='__main__':pass
