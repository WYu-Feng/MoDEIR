# Paper-Aligned MoDEIR

This is an isolated fork of `refined_modeir/`. The original project is left
unchanged; this folder adjusts only the implementation details that conflicted
with the camera-ready MoDEIR paper.

## Paper-Aligned Changes

- TSCM uses the paper Eq. (2) latent perturbation path by default:
  `--latent-mode strict_delta`.
- TAR uses top-s routing with `top_s=2` by default, and training enables
  Gumbel TopK sampling by default.
- Routed expert latents are fused by normalized router weights and then passed
  through a latent `ResBlock`, matching Eq. (1).
- FRM receives the largest selected timestep as its gate signal, matching
  Eq. (5), rather than a weighted-average timestep.
- FRM defaults to two stacked RRDB refiners and timestep-conditioned gating.
- The default training mode is `paper_stage2`: freeze the expert pool, TSCM,
  and UNet injectors; train TAR, latent fusion, and decoder FRMs.
- Stage 2 default losses follow the paper objective:
  image L1 + `0.1 * L_severity` + `0.2 * L_i-adv`.

All other project behavior is inherited from the current refined implementation,
including the Stable Diffusion backend, DA-CLIP router backbone, windowed
attention engineering, static TSCM injection blocks, checkpoint-compatible
loading, and portable vendor layout.

## Weights

`weights/legacy/` is hard-linked from `refined_modeir/weights/legacy/` to avoid
duplicating large checkpoint files. Treat these as read-only pretrained inputs.
New paper-aligned checkpoints are saved under this folder's own `outputs/`.

## Train Paper Stage 2

```bash
cd ~/modeir_refactor_project/modeir_paper_aligned
python train.py \
  --dataset-root /home/Baseline/ceshi/datasets \
  --output-dir outputs/paper_stage2
```

Important defaults:

- `--train-mode paper_stage2`
- `--train-size 192`
- `--batch-size 6`
- `--max-steps 30000`
- `--top-s 2`
- `--latent-mode strict_delta`
- `--router-gumbel-topk`
- `--target-per-task 10000`

The staged training launcher `scripts/train_extreme_stages.sh` checks the
required five-task folders before training:

- `Image deblurring/GoPro/test`
- `Image deblurring/GoPro/train`
- `Image dehazing/SOTS/outdoor`
- `Image denoise/BSD400`
- `Image denoise/BSD68`
- `Image deraining/RainTrainL`
- `Image deraining/Rain100L`
- `Low-light enhancement/LOL/eval`
- `Low-light enhancement/LOL/train`

Legacy ablation modes from the refined project remain available:
`router`, `experts`, `tscm_decoder`, `router_tscm_decoder`, and `joint`.

## Restore Images

```bash
python evaluate.py \
  --input-dir /path/to/degraded_images \
  --output-dir outputs/restored \
  --resume outputs/paper_stage2/refined_last.pt
```

Evaluation and paired testing also default to `strict_delta` and `top_s=2`.
