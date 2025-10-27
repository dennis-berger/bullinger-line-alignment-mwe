#!/usr/bin/env python3
# run_eval_qwen.py
"""
Bullinger MWE with Qwen3-VL:
- Walk data_val/gt/*.txt to get IDs
- Collect images in data_val/images/<ID>/** (supports multi-page)
- Transcribe with Qwen/Qwen3-VL-32B-Instruct (vision-language)
- Write predictions/<ID>.txt
- Compute WER/CER (raw + normalized) -> evaluation_qwen.csv

"""

import argparse
import csv
import glob
import os
import sys
from dataclasses import dataclass
from typing import List

from PIL import Image

# ---------------- Metrics (pure Python) ----------------
def _lev(a: list, b: list) -> int:
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
    return _lev(rt, ht) / max(1, len(rt))

def cer(ref: str, hyp: str) -> float:
    rc, hc = list(ref.strip()), list(hyp.strip())
    if not rc: return 0.0 if not hc else 1.0
    return _lev(rc, hc) / max(1, len(rc))

def normalize_whitespace(s: str) -> str:
    return " ".join(s.split())

# ---------------- Data helpers ----------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

def find_images_for_id(images_root: str, sample_id: str) -> List[str]:
    base = os.path.join(images_root, sample_id)
    if not os.path.isdir(base): return []
    cand = []
    # direct files
    for ext in IMG_EXTS:
        cand += sorted(glob.glob(os.path.join(base, f"*{ext}")))
    # common subfolders
    for sub in ("page", "images", "img"):
        d = os.path.join(base, sub)
        if os.path.isdir(d):
            for ext in IMG_EXTS:
                cand += sorted(glob.glob(os.path.join(d, f"*{ext}")))
    # last resort: recursive
    if not cand:
        for ext in IMG_EXTS:
            cand += sorted(glob.glob(os.path.join(base, "**", f"*{ext}"), recursive=True))
    # dedup keeping order
    seen, out = set(), []
    for p in cand:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def read_text(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip()

def write_text(p: str, t: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(t)

# ---------------- Qwen backend ----------------
@dataclass
class QwenCfg:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "auto"   # "auto"|"cuda"|"cpu"
    max_new_tokens: int = 2048

class QwenTranscriber:
    def __init__(self, cfg: QwenCfg):
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.torch = torch
        self.device = "cuda" if (cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
        # device_map="auto" spreads across available GPU RAM (important for big models)
        self.model = AutoModelForVision2Seq.from_pretrained(
            cfg.model_id,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        self.max_new_tokens = cfg.max_new_tokens

    def _prompt(self) -> str:
        # Keep instructions strict to avoid extra commentary
        return (
            "Transcribe the following handwritten page into plain text. "
            "Output ONLY the raw transcription. Preserve line breaks. "
            "Do not add explanations, translation, or extra characters."
        )

    def transcribe_images(self, image_paths: List[str]) -> str:
        from transformers import GenerationConfig
        texts = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            inputs = self.processor(images=img, text=self._prompt(), return_tensors="pt")
            # Move to the same device as model (for CPU this is a no-op)
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            gen_cfg = GenerationConfig(max_new_tokens=self.max_new_tokens)
            out_ids = self.model.generate(**inputs, generation_config=gen_cfg)
            text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            texts.append(text)
        return "\n".join(texts).strip()

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_val", help="folder containing gt/ and images/")
    ap.add_argument("--out-dir", default="predictions", help="where to write predictions")
    ap.add_argument("--eval-csv", default="evaluation_qwen.csv", help="output CSV path")
    ap.add_argument("--hf-model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--hf-device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    # Instantiate backend
    transcriber = QwenTranscriber(QwenCfg(
        model_id=args.hf_model,
        device=args.hf_device,
        max_new_tokens=args.max_new_tokens,
    ))

    gt_dir = os.path.join(args.data_dir, "gt")
    images_root = os.path.join(args.data_dir, "images")
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    if not gt_files:
        print(f"No ground-truth files found in {gt_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    n = 0
    sum_w = sum_c = sum_wn = sum_cn = 0.0

    for gt_path in gt_files:
        sample_id = os.path.splitext(os.path.basename(gt_path))[0]
        img_paths = find_images_for_id(images_root, sample_id)
        if not img_paths:
            print(f"[WARN] No images for {sample_id}; skipping.", file=sys.stderr)
            continue

        pred = transcriber.transcribe_images(img_paths)
        write_text(os.path.join(args.out_dir, f"{sample_id}.txt"), pred)

        gt = read_text(gt_path)
        w, c = wer(gt, pred), cer(gt, pred)
        wn, cn = wer(normalize_whitespace(gt), normalize_whitespace(pred)), cer(normalize_whitespace(gt), normalize_whitespace(pred))
        rows.append([sample_id, len(gt), len(pred), w, c, wn, cn])
        sum_w += w; sum_c += c; sum_wn += wn; sum_cn += cn; n += 1
        print(f"[OK] {sample_id}: WER={w:.3f} CER={c:.3f} (norm WER={wn:.3f} CER={cn:.3f})")

    # Write CSV (+ macro average)
    os.makedirs(os.path.dirname(args.eval_csv) or ".", exist_ok=True)
    with open(args.eval_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["id", "len_gt", "len_pred", "wer", "cer", "wer_norm", "cer_norm"])
        wtr.writerows(rows)
        if n > 0:
            wtr.writerow([])
            wtr.writerow(["macro_avg", "", "", sum_w/n, sum_c/n, sum_wn/n, sum_cn/n])

    print(f"\nWrote {args.eval_csv} with {n} samples.")

if __name__ == "__main__":
    main()
