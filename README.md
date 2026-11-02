# qwen3-8b-med6m — code, statistics and paper

Code and statistics for the LoRA adapter **Qwen3-8B med6m (S1_v2r, checkpoint 443, scale 0.5)** by Giovanni Del Papa (*Lo Scienziato Dilettante GDP*). Weights/model card: Hugging Face `odelpa/qwen3-8b-med6m`. DOI: [10.5281/zenodo.23895100](https://doi.org/10.5281/zenodo.23895100). Paper (PDF, rendered formulas, figures, tables): `paper/article_it.pdf`, `paper/article_en.pdf`.

> Not a medical device; no medical advice. Evaluation is multiple-choice letter log-probability only.

**Headline (sealed, single-use test, n=3766 medical items):** 61.34 % → 63.67 %, Δ = +2.34 pp, 95 % paired-bootstrap CI [+1.14, +3.53], exact McNemar p = 1.4·10⁻⁴ (306 vs 218). MedQA (+1.55) and MMLU-med (+0.74) are *not* individually significant. ARC-it guard +0.74 pp (no regression). Checkpoint 443, not the final 886.

$$W = W_0 + s\,\frac{\alpha}{r}\,BA,\qquad s=0.5,\ \alpha/r=1$$

$$\Delta_{\mathrm{pp}}=\frac{100\,(n_+-n_-)}{n},\qquad p=\min\Big\{1,\,2\sum_{k=0}^{\min(n_-,n_+)}\binom{m}{k}2^{-m}\Big\}$$

(If your viewer does not render math: `figures/eq_lora.png`, `eq_delta.png`, `eq_mcnemar.png`.)

## Reproduce the statistics (CPU, seconds)
```
pip install -r code/requirements.txt      # numpy, matplotlib
python code/recompute.py                  # -> results/recomputed.json (paired bootstrap B=20000 seed 20261101 + exact McNemar)
python code/make_figures.py               # -> figures/*.png|pdf
```
Inputs: `results/paired_counts.json` (per-item paired outcome counts, sealed test), `results/dev_final886_counts.json` (dev, not independent).

## Contents
`code/` statistics + figures · `results/` counts, recomputed stats, `training_config.json`, `CHECKSUMS.sha256` · `figures/` · `paper/` LaTeX sources + PDFs (build: `python paper/build.py article_en article_it`, needs `tectonic`) · `ATTRIBUTION.md` · `NETWORK_AUDIT.md` · `LICENSE` (Apache-2.0, code). The adapter and paper/figures are CC BY-SA 4.0 (see HF repo).

## Open items (honest status)
* Original training/eval scripts are in `code/original/` (verbatim copies from the training PC; they use local paths of that machine).
* GSM8K / IFEval on checkpoint 443: not run (`eval/PENDING_GSM8K_IFEVAL.md`).
* Dynamic network audit: not done (`NETWORK_AUDIT.md`).
* Pinned dataset revisions are the local HF cache revisions (Appendix A of the paper); the cached dates (24–30 Oct 2026) match the training window but exact use is not proven.
