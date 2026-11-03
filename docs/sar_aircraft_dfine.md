# D-FINE-M on SAR-AIRcraft-1.0

## Frozen baseline protocol (fixed 2026-11-03)

Config: `configs/dfine/sar_aircraft/dfine_hgnetv2_m_sar_aircraft_baseline.yml`. Module experiments inherit this file and change only the model.

| Item | Setting |
| --- | --- |
| Training data | **trainval** 3489 images (official train 3047 + val 442) |
| Evaluation | **test** 879 images, **evaluated once** with the final EMA weights after training |
| Model selection | **None.** Fixed schedule (solver `fixed_schedule: True`): no per-epoch eval, no best_stg1 reload / EMA rollback at stop_epoch |
| Init | `dfine_m_obj2coco_airplane7.pth` (Obj365→COCO D-FINE-M, 7-class score heads initialized from COCO airplane) |
| Input | `PadToCanvas 1500` → `Resize 1024` (one scale factor for every image, 0.683), the same at test time |
| Augmentation | RandomHorizontalFlip + batch multi-scale (768–1280, base repeated 6×). No ZoomOut / IoUCrop / PhotometricDistort / Mosaic / MixUp |
| Optimization | AdamW lr 1.75e-4 (backbone 1.75e-5), wd 1.25e-4, linear warmup 500 it, then constant LR; grad clip 0.1 |
| Batch | Global 16 (2 GPUs × 8), AMP, sync BN |
| Schedule | 120 ep = 112 multi-scale + 8 plain (EMA restart decay 0.9999); 218 it/ep, ~26.2k it total |
| EMA | decay 0.9999, warmups 1000 |
| seed | 0 |
| Metrics | COCO AP50:95, **AP50**, AP75, per-class AP50 (computed from `eval/final.pth`) |

Ablation configs use the same protocol: `..._resize1024.yml` (no scale preservation, plain resize as in PGD).

### Why the epoch count is fixed in advance

With trainval used for training, there is no held-out val set; looking at test to choose epochs would leak test information. So the schedule is set ahead of time from existing evidence:

- DEIM-M-1024 (train only, b12, quadratic warmup that wasted ~15 epochs) was still rising on val at ep 102, i.e. ~26k iterations.
- trainval adds 14.5% iterations per epoch.
- 120 ep ≈ 26.2k iterations, which errs slightly on the long side. In the DEIM-640 run, val stayed flat from ep 65 to 101 with no decline, so the risk of training too long is small.

Do not change the epoch count based on test results. If a schedule study is needed, run it in a train → val dev phase and then freeze it again.

## Comparing with PGD

PGD: Physics-Guided Detector for SAR Airplanes, TCSVT 2025, [arXiv 2411.12301](https://arxiv.org/abs/2411.12301), [code](https://github.com/XAI4SAR/PGD).

**Our protocol (trainval → test) is not the same as PGD's (most likely train → val), so the numbers cannot be compared directly.**
PGD reports YOLOv8-R50 88.7 and PGD-YOLOv8 90.7. Its DETR-family baselines (300 ep, run by the authors): DINO 85.2, DETReg 87.0, DEQDet 80.3.
When writing the paper, either state the difference in protocol or rerun the baselines under ours.

### Which split are PGD's numbers on? Evidence (checked 2026-11-03)

The paper only gives the split sizes and says training uses the "Training set". It never says which split the Table I/II numbers were measured on; the only explicit mention is Fig. 5's visualization ("on test data").
Code (XAI4SAR/PGD @94b712c, `train_PGD.py`):

- Trains on `data['train']` and validates on `data['val']` every epoch. `best.pt` is chosen by val fitness.
- After training it re-runs "Validating best.pt" **on val_loader**.
- The README has only training commands, nothing for evaluating on test.

Cross-check against our own data: DEIM-M (`102e_b28_nozoom_nocrop`), per-class AP50 (B787/A220/A320/B737/A330/ARJ21/other):

| | Per class | Mean |
| --- | --- | --- |
| PGD table, YOLOv8-R50 | 91.2 / 85.3 / 97.5 / 80.7 / 95.0 / 87.4 / 84.1 | 88.7 |
| Ours, val | 93.7 / 85.7 / 95.4 / 75.9 / 91.9 / 88.6 / 85.2 | 88.1 |
| Ours, test | 58.8 / 55.5 / 82.9 / 35.5 / 93.0 / 61.6 / 70.3 | 65.4 |

- PGD's per-class pattern matches our val closely and our test poorly. All 8 DEIM variants drop 21–25 AP50 from val to test.
- 7.2% of val images have a near-duplicate in train (similarity > 0.9), while test has 0% (`diag_test_eval_all`).
- If val had been used for training, it would be close to 100, so "trainval training + val reporting" is ruled out.
- External signal (weak): two non-maintainer PRs (#11, #12, 2026-10-18, identical diff, no maintainer reply) claim PGD-YOLOv5 gets 76.9% mAP@50 on test.
- Its † baselines are copied from SFS-CNet (CVPR 2024), which I couldn't access, so whether they share the same protocol is unconfirmed.

Conclusion: PGD most likely trained on train, selected on val and reported on val. The paper does not say this explicitly.

## SAR-specific choices and their evidence

1. **Scale-preserving input (`PadToCanvas 1500` → `Resize 1024`).**
   Native long-side medians are the same at every image size (A330 ≈ 120 px, A320 ≈ 100, B737 ≈ 91, ARJ21 ≈ 65), so the GSD is constant.
   A direct resize of the 800–1500 px images to one fixed size gives per-image scale factors 2× apart. After resize-to-640, an A220 from an 800 image (70 px) comes out larger than an A330 from a 1500 image (51 px).
   `PadToCanvas` pads on the right and bottom and sets `orig_size` to the canvas. Box round-trip error is 0 px. The gain still needs to be verified with `_resize1024`.
2. **RandomZoomOut and RandomIoUCrop off.** Both change object scale, which breaks point 1. ZoomOut also fills with zeros, which doesn't look like SAR clutter. The DEIM runs also turned them off.
3. **RandomPhotometricDistort off.** The images are single-channel (PIL `L`), so hue, saturation and channel shuffle do nothing. Contrast 0.5–1.5 rewrites the backscatter and speckle statistics.
4. **Horizontal flip kept.** PGD also uses fliplr 0.5. Aircraft are mirror-symmetric. No vertical flip or rotation, same as PGD.
5. **Batch multi-scale kept.** In the DEIM-M runs, configs identical except multi-scale on/off gave val AP 64.5 vs 58.7 (single seed). Multi-scale perturbs absolute scale by ±25%, which pulls against point 1, so it is the first thing to re-check.
6. Mosaic / MixUp not used. D-FINE's built-in Mosaic cannot be used here as is.

## Schedule and start-up tricks (based on the DEIM-M 640 curve)

**What makes the start slow: LR, not EMA.**
DEIM's warmup is quadratic: `lr = init × (it/2000)²`. 2000 it ≈ 18.5 ep, and at ep 9 the LR is only 26% of peak. Val AP is still 8.6 at ep 15, then goes 13 → 49 right after warmup ends.
Evaluating the raw `model` weights from the checkpoints separately:

| ep | 11 | 15 | 23 | 31 | 47 | 63 | 87 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EMA AP | 4.5 | 8.6 | 37.2 | 51.5 | 57.7 | 64.2 | 64.5 |
| raw AP | 4.6 | 8.1 | 36.4 | 50.2 | 56.8 | 62.9 | 63.0 |

The raw weights are just as slow, so the slow start is set by the LR. EMA is consistently +1.3–1.5 AP late in training, so keep it. The plateau after ep 65 is real convergence.

**Tricks:**

1. **Short linear warmup**: 500 it ≈ 2.3 ep, instead of DEIM's ~18.5-ep quadratic warmup.
2. **Initialize the score heads from COCO "airplane"** (`tools/dataset/init_head_from_coco_class.py`). D-FINE's 80→7 head mapping fails and falls back to random heads.
   3-epoch A/B, train → val, 1 GPU, b8, same seed:

   | ep | 0 | 1 | 2 |
   | --- | --- | --- | --- |
   | random head AP / AP50 | 0.1 / 0.3 | 2.3 / 4.9 | 3.4 / 7.1 |
   | airplane head AP / AP50 | 4.8 / 10.2 | 7.7 / 14.8 | 10.6 / 20.3 |

   Its effect on final AP has not been measured; it mainly speeds up the start.
3. **LR sqrt scaling**: D-FINE official 2.5e-4 @ b32 → 1.75e-4 @ b16, the same rule as the earlier DEIM runs.

## Resources

Smoke test (2026-11-03): single 4090, 8 images per GPU, AMP: peak **21.3 GB** (multi-scale up to 1280), 0.70 s/it.
Baseline 120 ep × 218 it: a real 2-GPU launch (2 × 8) measured 0.65 s/it with a peak of 21.3 GB per GPU, about **5 h** in total.
The server's GPUs are shared, so the launch script aborts if any selected GPU already uses more than 1 GB.

## Running

Generate the init checkpoint once (already done on the server, 2026-11-03):

```bash
python tools/dataset/init_head_from_coco_class.py --src /root/zxy/pretrain/dfine/dfine_m_obj2coco.pth \
  --dst /root/zxy/pretrain/dfine/dfine_m_obj2coco_airplane7.pth --num-classes 7 --coco-label 4
```

Start the baseline (output directory: `/root/zxy/experiments/<config name>_seed0_<time>`). The script records the git revision, git status, resolved config, launch command and nvidia-smi:

```bash
cd /root/zxy/D-FINE && git pull --ff-only
bash tools/train_sar_aircraft.sh configs/dfine/sar_aircraft/dfine_hgnetv2_m_sar_aircraft_baseline.yml 0,1
```

Resume after an interruption (from `last.pth`, which is saved every epoch):

```bash
bash tools/train_sar_aircraft.sh --resume /root/zxy/experiments/<run directory> 0,1
```

Progress: `tail -f $OUT/console.log`. `log.txt` holds per-epoch training losses only. The last line, `final_coco_eval_bbox[0/1/2]`, is test AP / AP50 / AP75.

Per-class test AP50:

```bash
python - "$OUT/eval/final.pth" <<'PY'
import sys, torch, numpy as np
p = np.asarray(torch.load(sys.argv[1], weights_only=False)["precision"])  # T,R,K,A,M
names = ["A220", "A320/321", "A330", "ARJ21", "Boeing737", "Boeing787", "other"]
ap = [(lambda x: x[x > -1].mean() * 100)(p[0, :, k, 0, -1]) for k in range(p.shape[2])]
print(" ".join(f"{n}={a:.1f}" for n, a in zip(names, ap)), f"| mAP50={np.mean(ap):.1f}")
PY
```

Re-evaluate the final weights (should match the last line of log.txt):

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/dfine/sar_aircraft/dfine_hgnetv2_m_sar_aircraft_baseline.yml \
  -r "$OUT/final.pth" --test-only --output-dir "$OUT/retest"
```

## Dev recipe (train → val)

`dfine_hgnetv2_m_sar_aircraft.yml` is the recipe file: train → val, 104 ep, with D-FINE's original best_stg selection. Use it only when a dev experiment needs val feedback, such as a schedule or hyperparameter study. Final numbers always come from the frozen protocol.
