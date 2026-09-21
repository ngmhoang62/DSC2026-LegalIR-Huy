HARRIER (LoRA vietlegal) - DIEM SO + SCRIPT + ADAPTER
=====================================================
Giai nen vao thu muc goc project (giu nguyen duong dan tuong doi).

FILE BAN YEU CAU
----------------
results/from_drive/harrier_ft_cv.pkl       diem CV 600 cau   <- score_cv_harrier_lora.py
results/from_drive/harrier_ft_public.pkl   diem public 1000  <- score_public_harrier_lora.py

KEM THEO (de tai tao duoc diem so)
----------------------------------
models/from_drive/vietlegal_finetuned_results_HNSW/vietlegal_finetuned/best_adapter/
    adapter LoRA r=16, che do FEATURE_EXTRACTION

KHONG KEM (qua nang, 3.6 GB - tai rieng neu can)
------------------------------------------------
models/vietlegal-harrier-0.6b/   base model: mainguyen9/vietlegal-harrier-0.6b

BOI CANH
--------
Day la kenh cua ban burst_jf_harrier_maxrecall (CV 0.9594 - CAO NHAT tung do).
Harrier dung rieng chi dat Recall@5 = 0.8817 (yeu nhat trong 3 model tu Drive),
nhung tren CV no trong nhu bo khuyet cho jina_ft (bat duoc 24 gold jina_ft bo sot).
TREN LEADERBOARD THAT NO THUA ban 0.9561 (aiteamvn_ft+jina_ft+title_embed).
Ly do: 600 cau CV khong du de phan biet bo khuyet that voi loi ngau nhien cua
mot model yeu; chenh lech 0.0033 nam duoi san nhieu 0.008.
