# Attribution / Attribuzione

**Adapter / adattatore:** `qwen3-8b-med6m` (LoRA r=64, checkpoint 443 of stage S1_v2r) — © 2026 Giovanni Del Papa, released under **CC BY-SA 4.0** (share-alike inherited from some training sources; see below). Code: Apache-2.0.

**Base model:** Qwen3-8B © Qwen Team / Alibaba Cloud, Apache-2.0 (<https://huggingface.co/Qwen/Qwen3-8B>); 4-bit bitsandbytes repack by Unsloth `unsloth/Qwen3-8B-unsloth-bnb-4bit` (Apache-2.0), revision `62efd7f9d748e394734a7adae2adf96e13a2abc8`.

| Training source | Licence | Role |
|---|---|---|
| MedMCQA train (`openlifescienceai/medmcqa`) | Apache-2.0 (declared; exam-question provenance not independently verified) | medical (42% of medical quota) |
| Wikipedia IT medical subset (`wikimedia/wikipedia` 20231101.it) | CC BY-SA 3.0 / GFDL — © Wikipedia contributors | medical (32% of medical quota) |
| Wikipedia EN medical subset (20231101.en) | CC BY-SA 3.0 / GFDL — © Wikipedia contributors | medical (21% of medical quota) |
| MedQA train (`openlifescienceai/medqa`, mirror of jind11/MedQA, train split only) | MIT | medical (5% of medical quota) |
| OASST2 (`OpenAssistant/oasst2`) | Apache-2.0 | replay |
| Aya dataset (`CohereLabs/aya_dataset`) | Apache-2.0 | replay |
| Dolly 15k (`databricks/databricks-dolly-15k`) | CC BY-SA 3.0 | replay |
| Wikipedia IT general | CC BY-SA 3.0 / GFDL | replay |
| FineWeb-Edu, FineWeb-2 (ita_Latn) | ODC-By 1.0 | replay |

Evaluation data (not used for training): MedMCQA validation, MedQA-USMLE-4-options test (`GBaker`), MMLU (`cais/mmlu`, medical subdomains), PubMedQA (`qiaojin/PubMedQA`, pqa_labeled), ARC-Italian (`sapienzanlp/arc_italian`, CC BY-SA 4.0). Pinned dataset revisions are in `paper/article_en.pdf`, Appendix A.

No non-commercial (NC) or no-derivatives (ND) source was used. All sources allow commercial use. Under CC BY-SA, derivative material must carry the same licence: the adapter is therefore released as CC BY-SA 4.0 (3.0 → 4.0 upgrade is allowed by the CC BY-SA 3.0 text for "later versions with the same elements"). *Whether share-alike legally extends to model weights is unsettled; this is a conservative choice and should be confirmed by a legal reading before publication.*
