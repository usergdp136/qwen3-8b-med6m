# Network audit note — med6m S1_v2r (rule of 01/11/2026)

**Date of this note:** 2 Nov 2026 (Europe/Rome). **Type:** static review + live check of the training machine. **Result: no unauthorised destination found; dynamic (traffic-capture) audit NOT done.**

Static review of `build_s1v2r.py`, `train_stage_r.py`, `eval_mc.py`, `s0_data.py`, `s0b_data.py`, `newsrc_build/small.py`, `evalgrams.py`:
* only `datasets.load_dataset` / `from_pretrained` calls to Hugging Face repos (listed in ATTRIBUTION.md); no `requests`/`urlopen` in training or evaluation;
* `SFTConfig(report_to='none')` — no Weights & Biases / TensorBoard uploads;
* `pmc.py` (Europe PMC open-access S3 bucket `pmc-oa-opendata.s3.amazonaws.com`) is a data-preparation helper, not used by training/evaluation of S1_v2r;
* training and scoring read local JSONL (`train.jsonl`, `eval/*.jsonl`).

**Not verified / residual risk:** no `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE` or `UNSLOTH_DISABLE_STATISTICS` flag found in the launch scripts, so library-level telemetry (Hugging Face Hub metadata requests, Unsloth statistics ping) cannot be excluded for past runs. The machine hosts other processes; a past-run audit is impossible retroactively.

**Recommended closing step (CPU/GPU-light, ~30 min):** load base+adapter on the MSI or Everest with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UNSLOTH_DISABLE_STATISTICS=1`, run 50 dev items, capture connections (`Get-NetTCPConnection` / Wireshark filter on the python PID) and attach the log here.

## Log of network use for publication (02/11/2026, Europe/Rome)
Authorised by the author (20:24). Because the `hf` token configured on the training PC is read-only (403 on repo creation), the 698 MB adapter was split into eight chunks on the training PC, transferred to the agent box with the agent file-copy tool, re-assembled and verified (sha256 0dfdebe5…c4a5bf identical), and uploaded from the box to Hugging Face with the already-configured write credential; the GitHub push and the Zenodo deposit were also made from the box. Training/evaluation code was not run. These are deliberate outbound transfers of public artefacts, logged here and in the second-brain activity log.
