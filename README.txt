BUNG CAC ARTIFACT CUA KENH `base` + SCRIPT SINH RA CHUNG
=========================================================
Giai nen vao thu muc goc cua project (giu nguyen duong dan tuong doi).

ANH XA ARTIFACT -> SCRIPT TAO RA
--------------------------------
results/jina_reranker/burst_pairwise_state.pt      <- finetune_jina_burst.py        (dong 198)
    (kem manifest burst_pairwise_training.json)
results/burst_large_ltr/best_model.pkl             <- tune_burst_large_ltr.py       (dong 189)
results/burst_legal_features/validation_model.pkl  <- tune_burst_legal_features.py  (dong 190)
results/burst_empirical_pairwise/model.pkl         <- tune_burst_empirical_pairwise.py (dong 158)
results/burst_gpu_threeview/cpu_top20.pkl          <- run_burst_gpu_submission.py   (dong 97-140)

THU TU DUNG LAI
---------------
1. tune_burst_large_ltr.py          -> best_model.pkl        (XGBoost; can cache retrieval SQLite-FTS)
2. tune_burst_legal_features.py     -> validation_model.pkl  (can cache retrieval + corpus da tokenize, ~48 phut CPU)
3. tune_burst_empirical_pairwise.py -> model.pkl             (dung cache retrieval cua burst_large_ltr)
4. finetune_jina_burst.py           -> burst_pairwise_state.pt (~48 giay GPU; GHI DE, khong co cache)
5. run_burst_gpu_submission.py      -> cpu_top20.pkl         (hop 4 nhanh bang weighted_rrf; co cache theo CPU_CONFIG)

CANH BAO
--------
Ca 5 artifact deu nam trong chuoi dung ban 0.9561 (results/burst_userft_maxrecall/submission.zip).
Chay lai bat ky script nao o tren se GHI DE artifact tuong ung; rieng finetune_jina_burst.py
khong kiem tra file ton tai. Sao luu truoc khi chay lai.
