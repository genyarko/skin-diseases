"""Multi-model ensemble + proper TTA — averages softmax across 1+ checkpoints.

Usage:
  python ensemble_eval.py best_ema.pt
  python ensemble_eval.py best_ema_v2.pt convnextv2_best_ema.pt
  python ensemble_eval.py best_ema_v2.pt swa_v2.pt convnextv2_best_ema.pt convnextv2_swa.pt

Each checkpoint is loaded with its own model_name + img_size from its dict
(so EVA-02 @ 448 and ConvNeXt V2 @ 384 mix freely). Each runs TTA = identity +
hflip + vflip + 2 scale crops, then per-image softmaxes are averaged.

Saves predictions.csv and prints classification_report.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import timm
import torch
from sklearn.metrics import classification_report, f1_score
from timm.data import create_transform
from torch.utils.data import DataLoader
from torchvision import datasets

BASE_EXTRACT_DIR = "./dermnet-skin40-cleaned-dataset"
DATA_DIR = os.path.join(BASE_EXTRACT_DIR, "kaggle/working/Merged_Dermnet_Skin40")
TEST_DIR = os.path.join(DATA_DIR, "test")
BATCH_SIZE = 16
NUM_WORKERS = 4
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

os.environ["HIP_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(img_size, crop_pct):
    """Each (img_size, crop_pct) combo gets its own loader for proper TTA."""
    tf = create_transform(input_size=img_size, is_training=False,
        crop_pct=crop_pct, interpolation="bicubic", mean=MEAN, std=STD)
    ds = datasets.ImageFolder(TEST_DIR, transform=tf)
    return ds, DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = ckpt["model_name"]
    img_size = ckpt["img_size"]
    sd = ckpt["model_state_dict"]
    # Infer num_classes from head weight
    head_keys = [k for k in sd.keys() if k.endswith("head.weight") or k.endswith("fc.weight")]
    num_classes = sd[head_keys[0]].shape[0] if head_keys else 23
    model = timm.create_model(model_name, pretrained=False,
        num_classes=num_classes, img_size=img_size if "vit" in model_name or "eva" in model_name else None)
    if "img_size" in timm.create_model.__code__.co_varnames:
        pass
    # Some non-ViT models reject img_size kwarg — handle gracefully
    try:
        model = timm.create_model(model_name, pretrained=False,
            num_classes=num_classes, img_size=img_size)
    except TypeError:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(sd)
    model = model.to(device, memory_format=torch.channels_last).eval()
    print(f"  loaded {ckpt_path}  ({model_name} @ {img_size}, "
          f"prev acc={ckpt.get('val_acc', 0)*100:.2f}%)")
    return model, img_size


@torch.no_grad()
def tta_softmax(model, img_size):
    """4-augmentation TTA: identity + hflip @ crop_pct=0.95, plus same pair @ crop_pct=1.0."""
    aggregated = None
    targets = None
    classes = None

    for crop_pct in [0.95, 1.0]:
        ds, loader = make_loader(img_size, crop_pct)
        if classes is None:
            classes = ds.classes

        batch_softmax = []
        batch_targets = []
        for x, y in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # identity
                p = torch.softmax(model(x), dim=-1).float()
                # hflip
                p = p + torch.softmax(model(torch.flip(x, dims=[-1])), dim=-1).float()
            batch_softmax.append(p.cpu())
            batch_targets.append(y)

        crop_softmax = torch.cat(batch_softmax)  # 2 augs already summed
        if aggregated is None:
            aggregated = crop_softmax
            targets = torch.cat(batch_targets)
        else:
            aggregated = aggregated + crop_softmax

    # Total: 2 crops × 2 flips = 4 augmentations summed
    aggregated = aggregated / 4.0
    return aggregated, targets, classes


def main():
    if len(sys.argv) < 2:
        print("Usage: python ensemble_eval.py <ckpt1> [ckpt2 ...]")
        sys.exit(1)

    ckpts = sys.argv[1:]
    print(f"Ensembling {len(ckpts)} checkpoint(s) with TTA (4 augs/model):")

    all_probs = []
    targets = None
    classes = None

    for ckpt in ckpts:
        if not os.path.exists(ckpt):
            print(f"  SKIP missing: {ckpt}"); continue
        model, img_size = load_model(ckpt)
        probs, t, cls = tta_softmax(model, img_size)
        all_probs.append(probs)
        if targets is None:
            targets, classes = t, cls
        # Free GPU memory between models
        del model; torch.cuda.empty_cache()

    if not all_probs:
        print("No valid checkpoints loaded."); sys.exit(1)

    # Equal-weight average across models
    ensemble = torch.stack(all_probs).mean(0)
    preds = ensemble.argmax(1)

    acc = (preds == targets).float().mean().item()
    f1 = f1_score(targets.numpy(), preds.numpy(), average="macro")

    print(f"\n{'='*60}")
    print(f"Ensemble of {len(all_probs)} model(s) + 4-aug TTA")
    print(f"{'='*60}")
    print(f"Accuracy:   {acc*100:.2f}%")
    print(f"Macro F1:   {f1:.4f}")
    print(f"\nPer-class report:")
    print(classification_report(targets.numpy(), preds.numpy(),
        target_names=classes, digits=3, zero_division=0))

    # Save predictions for further analysis
    np.savetxt("predictions.csv",
        np.column_stack([targets.numpy(), preds.numpy(), ensemble.numpy()]),
        delimiter=",", fmt="%g",
        header="true,pred," + ",".join(f"p_{c}" for c in classes), comments="")
    print(f"Saved predictions.csv  ({ensemble.shape[0]} rows)")


if __name__ == "__main__":
    main()
