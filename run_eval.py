#!/usr/bin/env python3
# run_eval.py
"""
Minimal pipeline for Bullinger MWE:
- Walk data_val/gt/*.txt to get IDs
- Find images in data_val/images/<ID>/
- Transcribe with Hugging Face TrOCR (or Tesseract)
- Write predictions to predictions/<ID>.txt
- Compute WER/CER and line-level accuracy (forward + reverse) and write evaluation.csv

Backends:
  - hf_trocr (default): Hugging Face VisionEncoderDecoder (TrOCR)
  - tesseract: uses pytesseract + system tesseract

Examples:
  python run_eval.py --data-dir data_val                 # default hf_trocr
  python run_eval.py --backend hf_trocr --hf-model microsoft/trocr-base-handwritten
  python run_eval.py --backend tesseract
"""

import argparse
import csv
import glob
import io
import os
import sys
from dataclasses import dataclass
from typing import List

from PIL import Image
from metrics import (
    line_accuracy,
    line_accuracy_norm,
    reverse_line_accuracy,
    reverse_line_accuracy_norm,
)

# ---------------- Metrics ----------------
def _levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]

def wer(ref: str, hyp: str) -> float:
    rt, ht = ref.strip().split(), hyp.strip().split()
    if not rt: return 0.0 if not ht else 1.0
    return _levenshtein(rt, ht) / max(1, len(rt))

def cer(ref: str, hyp: str) -> float:
    rc, hc = list(ref.strip()), list(hyp.strip())
    if not rc: return 0.0 if not hc else 1.0
    return _levenshtein(rc, hc) / max(1, len(rc))

# ---------------- IO helpers ----------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

def find_images_for_id(images_root: str, sample_id: str) -> List[str]:
    base = os.path.join(images_root, sample_id)
    if not os.path.isdir(base): return []
    cand = []
    for ext in IMG_EXTS:
        cand += sorted(glob.glob(os.path.join(base, f"*{ext}")))
    for sub in ["page", "images", "img"]:
        d = os.path.join(base, sub)
        if os.path.isdir(d):
            for ext in IMG_EXTS:
                cand += sorted(glob.glob(os.path.join(d, f"*{ext}")))
    if not cand:
        for ext in IMG_EXTS:
            cand += sorted(glob.glob(os.path.join(base, "**", f"*{ext}"), recursive=True))
    seen, out = set(), []
    for p in cand:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def load_image_resized(path: str, max_side: int = 1600) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
    if scale > 1.0:
        im = im.resize((int(w / scale), int(h / scale)))
    return im

def read_text(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip()

def write_text(p: str, t: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(t)

def normalize(s: str) -> str:
    return " ".join(s.split())

# ---------------- Backends ----------------
@dataclass
class BackendConfig:
    name: str
    hf_model: str = "microsoft/trocr-base-handwritten"
    hf_device: str = "auto"  # "auto" | "cpu" | "cuda"

class Transcriber:
    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        if cfg.name == "hf_trocr":
            try:
                import torch
                from transformers import VisionEncoderDecoderModel, AutoProcessor
            except Exception as e:
                print("ERROR: Need `torch` and `transformers` for hf_trocr.", file=sys.stderr)
                raise e
            # lazy init (only once)
            self._torch = __import__("torch")
            self._processor = __import__("transformers").AutoProcessor.from_pretrained(cfg.hf_model)
            self._model = __import__("transformers").VisionEncoderDecoderModel.from_pretrained(cfg.hf_model)

            # choose device
            if cfg.hf_device == "cpu":
                self._device = "cpu"
            elif cfg.hf_device == "cuda":
                self._device = "cuda" if self._torch.cuda.is_available() else "cpu"
            else:  # auto
                self._device = "cuda" if self._torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            # Decoding settings (greedy decoding is fine for MWE)
            self._gen_kwargs = {"max_new_tokens": 1024}
        elif cfg.name == "tesseract":
            try:
                import pytesseract  # noqa: F401
            except Exception as e:
                print("ERROR: Backend 'tesseract' requires `pytesseract` + system tesseract.", file=sys.stderr)
                raise e
        else:
            raise ValueError(f"Unknown backend: {cfg.name}")

    def transcribe_images(self, image_paths: List[str]) -> str:
        if self.cfg.name == "hf_trocr":
            return self._hf_trocr(image_paths)
        elif self.cfg.name == "tesseract":
            return self._tesseract(image_paths)

    def _hf_trocr(self, image_paths: List[str]) -> str:
        # Process images individually and join with newlines (preserves page order)
        outputs = []
        for p in image_paths:
            img = load_image_resized(p, max_side=1600)
            pixel_values = self._processor(images=img, return_tensors="pt").pixel_values.to(self._device)
            pred_ids = self._model.generate(pixel_values, **self._gen_kwargs)
            text = self._processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            outputs.append(text.strip())
        return "\n".join(outputs).strip()

    def _tesseract(self, image_paths: List[str]) -> str:
        import pytesseract
        out = []
        for p in image_paths:
            im = Image.open(p).convert("L")
            # German letters; add 'eng' as fallback
            txt = pytesseract.image_to_string(im, lang="deu+eng")
            out.append(txt.strip())
        return "\n".join(out).strip()

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_val")
    ap.add_argument("--out-dir", default="predictions")
    ap.add_argument("--backend", choices=["hf_trocr", "tesseract"], default="hf_trocr")
    ap.add_argument("--hf-model", default="microsoft/trocr-base-handwritten",
                    help="Only for --backend hf_trocr (e.g., microsoft/trocr-large-handwritten)")
    ap.add_argument("--hf-device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Pick device for hf_trocr")
    ap.add_argument("--eval-csv", default="evaluation.csv")
    args = ap.parse_args()

    cfg = BackendConfig(name=args.backend, hf_model=args.hf_model, hf_device=args.hf_device)
    transcriber = Transcriber(cfg)

    gt_dir = os.path.join(args.data_dir, "gt")
    images_root = os.path.join(args.data_dir, "images")
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    if not gt_files:
        print(f"No ground-truth files found in {gt_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    sum_wer = sum_cer = sum_wn = sum_cn = 0.0
    sum_la = sum_lan = sum_rla = sum_rlan = 0.0
    n = 0

    for gt_path in gt_files:
        sample_id = os.path.splitext(os.path.basename(gt_path))[0]
        img_paths = find_images_for_id(images_root, sample_id)
        if not img_paths:
            print(f"[WARN] No images for {sample_id}; skipping.", file=sys.stderr)
            continue

        pred = transcriber.transcribe_images(img_paths)
        pred_path = os.path.join(args.out_dir, f"{sample_id}.txt")
        write_text(pred_path, pred)

        gt = read_text(gt_path)
        w, c = wer(gt, pred), cer(gt, pred)
        wn, cn = wer(normalize(gt), normalize(pred)), cer(normalize(gt), normalize(pred))
        la  = line_accuracy(gt, pred)
        lan = line_accuracy_norm(gt, pred)
        rla = reverse_line_accuracy(gt, pred)
        rlan = reverse_line_accuracy_norm(gt, pred)

        rows.append([sample_id, len(gt), len(pred), w, c, wn, cn, la, lan, rla, rlan])
        sum_wer += w; sum_cer += c; sum_wn += wn; sum_cn += cn
        sum_la += la; sum_lan += lan; sum_rla += rla; sum_rlan += rlan
        n += 1
        print(
            f"[OK] {sample_id}: "
            f"WER={w:.3f} CER={c:.3f} "
            f"(norm WER={wn:.3f} CER={cn:.3f}) "
            f"LineAcc={la:.3f} LineAcc_norm={lan:.3f} "
            f"RevLineAcc={rla:.3f} RevLineAcc_norm={rlan:.3f}"
        )

    if n > 0:
        os.makedirs(os.path.dirname(args.eval_csv) or ".", exist_ok=True)
        with open(args.eval_csv, "w", newline="", encoding="utf-8") as f:
            wtr = csv.writer(f)
            wtr.writerow([
                "id",
                "len_gt",
                "len_pred",
                "wer",
                "cer",
                "wer_norm",
                "cer_norm",
                "line_acc",
                "line_acc_norm",
                "rev_line_acc",
                "rev_line_acc_norm",
            ])
            wtr.writerows(rows)
            wtr.writerow([])
            wtr.writerow([
                "macro_avg",
                "",
                "",
                sum_wer/n,
                sum_cer/n,
                sum_wn/n,
                sum_cn/n,
                sum_la/n,
                sum_lan/n,
                sum_rla/n,
                sum_rlan/n,
            ])
        print(f"\nWrote {args.eval_csv} with {n} samples.")
    else:
        print("No evaluated samples.", file=sys.stderr)

if __name__ == "__main__":
    main()
