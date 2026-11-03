"""
Build a fine-tuning checkpoint whose N-class score heads start from one COCO class.

For SAR-AIRcraft all 7 classes are aircraft, so every row of enc_score_head /
dec_score_head.* / denoising_class_embed is copied from COCO "airplane" (label 4).
Without this, D-FINE's 80->N head mapping fails and the heads are random, so query
selection is random at the start of training. Rows diverge once per-class gradients flow.

Usage:
    python tools/dataset/init_head_from_coco_class.py \
        --src /root/zxy/pretrain/dfine/dfine_m_obj2coco.pth \
        --dst /root/zxy/pretrain/dfine/dfine_m_obj2coco_airplane7.pth \
        --num-classes 7 --coco-label 4
"""

import argparse
import re

import torch

HEAD_PATTERN = re.compile(r"^decoder\.(enc_score_head|dec_score_head\.\d+)\.(weight|bias)$")
DN_EMBED = "decoder.denoising_class_embed.weight"


def convert(state_dict, num_classes, coco_label):
    out, changed = dict(state_dict), []
    for k, v in state_dict.items():
        if HEAD_PATTERN.match(k):
            out[k] = v[coco_label : coco_label + 1].repeat(num_classes, *[1] * (v.dim() - 1)).clone()
            changed.append(k)
        elif k == DN_EMBED:
            # rows 0..C-1 are classes, last row is the padding (index = num_classes) entry
            rows = [v[coco_label : coco_label + 1].repeat(num_classes, 1), v[-1:]]
            out[k] = torch.cat(rows, 0).clone()
            changed.append(k)
    return out, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--coco-label", type=int, default=4, help="contiguous COCO label, airplane=4")
    args = parser.parse_args()

    state = torch.load(args.src, map_location="cpu", weights_only=False)
    new = {}
    if "ema" in state:
        module, changed = convert(state["ema"]["module"], args.num_classes, args.coco_label)
        new["ema"] = {"module": module}
    if "model" in state:
        new["model"], changed = convert(state["model"], args.num_classes, args.coco_label)
    torch.save(new, args.dst)
    print(f"{len(changed)} tensors re-initialised from COCO label {args.coco_label}:")
    for k in changed:
        print("  ", k)


if __name__ == "__main__":
    main()
