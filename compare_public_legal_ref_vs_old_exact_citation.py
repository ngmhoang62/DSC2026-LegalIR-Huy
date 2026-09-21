#!/usr/bin/env python
import json
from pathlib import Path

ROOT = Path(r"D:\Study\DSC2026\sota")

new_actions = ROOT / "results/manual/huy_public_legal_ref_challenger_certificate_rescue_v1/PUBLIC_ACTIONS_LABEL_FREE.json"
new_sub = ROOT / "results/manual/huy_public_legal_ref_challenger_certificate_rescue_v1/submission.json"

old_actions = ROOT / "results/gemini/huy_d1_exact_citation_public_v1/PUBLIC_CITATION_ACTIONS.json"
old_sub = ROOT / "results/gemini/huy_d1_exact_citation_public_v1/CANDIDATE_D1_EXACT_CITATION.json"

d1_path = ROOT / "results/gemini/huy_vnlegal_rank_ablation_v1/CANDIDATE_D1_VNLEGAL_SCORE_ONLY.json"

def load(p):
    return json.loads(p.read_text(encoding="utf-8"))

na = load(new_actions)
ns = load(new_sub)
d1 = load(d1_path)

new_action_map = na.get("actions", {})
print("NEW ACTIONS:", len(new_action_map))
for q, a in new_action_map.items():
    print("\nNEW qid:", q)
    print("  type:", a.get("addition_type"))
    print("  relation_family:", a.get("relation_family"))
    print("  relation_direction:", a.get("relation_direction"))
    print("  anchor_ref:", a.get("anchor_ref"))
    print("  anchor_doc:", a.get("anchor_doc"))
    print("  defender:", a.get("defender"))
    print("  challenger:", a.get("challenger"))
    print("  defender_rel:", a.get("defender_rel"))
    print("  challenger_rel:", a.get("challenger_rel"))

if old_actions.is_file() and old_sub.is_file():
    oa = load(old_actions)
    osub = load(old_sub)
    old_map = oa.get("actions", {})
    print("\nOLD EXACT-CITATION ACTIONS:", len(old_map))
    for q, a in old_map.items():
        print("OLD qid:", q)
        print("  defender:", a.get("defender_doc_id"))
        print("  challenger:", a.get("anchor_doc_id"))

    same_sub = ns == osub
    print("\nFULL SUBMISSION IDENTICAL TO OLD EXACT-CITATION:", same_sub)

    new_q = set(new_action_map)
    old_q = set(old_map)
    print("new_qids:", sorted(new_q))
    print("old_qids:", sorted(old_q))
    print("same_qids:", new_q == old_q)

    if len(new_q) == len(old_q) == 1 and new_q == old_q:
        q = next(iter(new_q))
        new_pair = (
            new_action_map[q].get("defender"),
            new_action_map[q].get("challenger"),
        )
        old_pair = (
            old_map[q].get("defender_doc_id"),
            old_map[q].get("anchor_doc_id"),
        )
        print("same replacement pair:", new_pair == old_pair)
        print("new pair:", new_pair)
        print("old pair:", old_pair)
else:
    print("\nOld exact-citation artifacts not found; cannot compare directly.")

# Sanity: show actual diff vs exact D1.
changed = []
for q, row in ns.items():
    base = d1[q]["answer"]
    if row["answer"] != base:
        changed.append((q, base, row["answer"]))
print("\nDIFF VS D1:", len(changed))
for q, before, after in changed:
    print(" ", q, before, "->", after)
