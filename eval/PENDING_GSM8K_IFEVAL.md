# Pending: GSM8K and IFEval on checkpoint 443 (scale 0.5)

Previous adapters of this project were checked for non-regression on GSM8K and IFEval. For med6m ck443 these have **not** been run. Dev files exist on the training PC: `D:\tuning\med6m\eval\dev\gsm8k.jsonl`, `ifeval.jsonl`.

Needs a generation harness (eval_mc.py is letter-logit only), 4-bit Qwen3-8B + adapter at scale 0.5, greedy decoding, thinking off (settings to match the earlier Qwen3-14B v2 evaluation); ~1–2 h on a T4 / ~30 min on a RTX 4080. Not started because the T4 is occupied by another job and the adapter (698 MB) cannot be moved to the MSI via the agent's file-copy tool.
Pass criterion proposed: no drop larger than 1 pp vs base on either (to be confirmed by Giovanni); report CIs.
