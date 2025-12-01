#!/usr/bin/env python3
# run_eval_qwen_m1.py
"""
Method 1: Bullinger with Qwen3-VL (image(s) + transcription):

- Walk data_val/gt/*.txt to get IDs (ground-truth line-broken text).
- For each ID:
  - Load page image(s) from data_val/images/<ID>/**.
  - Load the CORRECT transcription (no line breaks) from
    data_val/transcription/<ID>.txt.
  - Split the transcription across pages (heuristic, by character length).
- For each page i: send (image_i, chunk_i) to Qwen to only insert line breaks.
- Concatenate all page-level outputs → prediction for that letter.
- Evaluate vs data_val/gt/<ID>.txt:
    - WER / CER (raw + whitespace-normalized)
    - line-level accuracy (forward + reverse, raw + normalized)
- Write predictions_m1/<ID>.txt and evaluation_qwen_m1.csv.
"""

import argparse
import csv
import glob
import os
import sys
from dataclasses import dataclass
from typing import List

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from metrics import (
    wer,
    cer,
    normalize_whitespace,
    line_accuracy,
    line_accuracy_norm,
    reverse_line_accuracy,
    reverse_line_accuracy_norm,
)

# ---------------- Data helpers ----------------

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

def find_images_for_id(images_root: str, sample_id: str) -> List[str]:
    """Find all images belonging to a sample ID."""
    base = os.path.join(images_root, sample_id)
    if not os.path.isdir(base):
        return []
    cand: List[str] = []

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
            seen.add(p)
            out.append(p)
    return out

def read_text(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip()

def write_text(p: str, t: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(t)

# ---------------- Qwen backend (Method 1, multi-page) ----------------

@dataclass
class QwenCfg:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "auto"   # "auto" | "cuda" | "cpu"
    max_new_tokens: int = 800

class QwenLineBreaker:
    """
    Use Qwen3-VL to insert line breaks into a given correct transcription
    based on the visual layout of a multi-page letter.

    - We have one transcription string for the whole letter.
    - We have 1..N page images.
    - We split the transcription into N chunks (roughly equal-length in chars,
      on word boundaries) and process each (image_i, chunk_i) pair separately.
    """

    def __init__(self, cfg: QwenCfg):
        self.device = "cuda" if (cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)

        load_kwargs = dict(trust_remote_code=True)
        if self.device == "cuda":
            # Prefer 4-bit quantization to fit on 32GB GPUs
            try:
                load_kwargs.update({
                    "device_map": "auto",
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                })
            except Exception:
                # Fallback to fp16 if bitsandbytes not available
                load_kwargs.update({
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                })

        self.model = AutoModelForVision2Seq.from_pretrained(cfg.model_id, **load_kwargs)
        self.model.eval()
        self.max_new_tokens = cfg.max_new_tokens

    # ---------- Prompt construction ----------

    def _build_prompt(self, transcription: str) -> str:
        """
        Build an instruction prompt that explains that the transcription is correct
        and the model must only insert newline characters.
        """
        return (
            "You see a scanned page of a historical handwritten letter.\n\n"
            "Below is the CORRECT diplomatic transcription of the text on this page, "
            "but without line breaks matching the page layout:\n\n"
            "TRANSCRIPTION (single block):\n"
            f"{transcription}\n\n"
            "Your task:\n"
            "1. Insert newline characters so that the lines correspond to the line breaks "
            "   visible in the image.\n"
            "2. You may insert newline characters either between words OR inside words, "
            "   if the line in the image breaks in the middle of a word.\n"
            "3. Do NOT change, remove, or add any characters (letters, punctuation, accents, etc.).\n"
            "   Do NOT insert hyphen characters at line breaks; if the page splits a word, "
            "   just split the word across two lines without adding '-'.\n"
            "4. Preserve the exact order and spelling of all characters.\n"
            "5. Output ONLY the re-formatted transcription with newline characters, "
            "   no explanations or extra text."
        )


    # ---------- Image helper ----------

    def _downscale(self, img: Image.Image, max_side: int = 1280) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        if s <= max_side:
            return img
        scale = max_side / float(s)
        return img.resize((int(w * scale), int(h * scale)))

    # ---------- Transcription splitting across pages ----------

    def _split_transcription_across_pages(self, transcription: str, num_pages: int) -> List[str]:
        """
        Split the full transcription into `num_pages` contiguous chunks
        of roughly equal character length, respecting word boundaries.

        Heuristic: we don't know the true page boundaries, but this gives each
        page its own sub-transcription to format.
        """
        text = transcription.strip()
        if num_pages <= 1 or not text:
            return [text]

        words = text.split()
        if not words:
            return [text]

        # total chars including single spaces between words
        total_len = sum(len(w) for w in words) + (len(words) - 1)
        remaining_len = total_len
        remaining_pages = num_pages

        chunks: List[str] = []
        cur_words: List[str] = []
        cur_len = 0

        target = remaining_len / remaining_pages  # target chars for current chunk

        for w in words:
            add_len = len(w) + (1 if cur_words else 0)

            # If we already have some content and adding this word would push us
            # over the target, and we still need pages after this, cut here.
            if cur_words and (cur_len + add_len > target) and (remaining_pages > 1):
                chunks.append(" ".join(cur_words).strip())
                remaining_len -= cur_len
                remaining_pages -= 1
                target = remaining_len / remaining_pages
                cur_words = [w]
                cur_len = len(w)
            else:
                cur_words.append(w)
                cur_len += add_len

        if cur_words:
            chunks.append(" ".join(cur_words).strip())

        # If we produced fewer chunks than pages (e.g. very short text),
        # pad with empty strings so zip(image_paths, chunks) still matches.
        while len(chunks) < num_pages:
            chunks.append("")

        return chunks

    # ---------- Core generation ----------

    @torch.inference_mode()
    def _generate_one(self, img: Image.Image, transcription: str) -> str:
        """
        Single-page call: image + transcription chunk -> line-broken chunk.
        """
        prompt = self._build_prompt(transcription)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            repetition_penalty=1.05,
        )
        raw = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        # --- Extract only the assistant part ---
        cleaned = raw.strip()
        marker = "\nassistant\n"
        idx = cleaned.rfind(marker)
        if idx != -1:
            cleaned = cleaned[idx + len(marker):].strip()

        # just in case it starts with a bare 'assistant' token
        if cleaned.startswith("assistant"):
            cleaned = cleaned[len("assistant"):].lstrip()

        return cleaned


    def infer_line_breaks(self, image_paths: List[str], transcription: str) -> str:
        """
        Method 1 core: use all page images.

        - Split the full transcription into N page chunks.
        - For each page i, run Qwen on (image_i, chunk_i).
        - Concatenate all page-level outputs into one prediction.
        """
        if not image_paths:
            raise ValueError("No image paths provided to QwenLineBreaker.")

        num_pages = len(image_paths)
        chunks = self._split_transcription_across_pages(transcription, num_pages)

        outputs: List[str] = []
        for img_path, chunk in zip(image_paths, chunks):
            if not chunk.strip():
                continue

            img = Image.open(img_path).convert("RGB")
            img = self._downscale(img, max_side=1280)

            out = self._generate_one(img, chunk)
            outputs.append(out.strip())

            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Join page outputs with a single newline between pages.
        return "\n".join(o for o in outputs if o).strip()

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_val",
                    help="Folder containing gt/, images/, transcription/")
    ap.add_argument("--out-dir", default="predictions_m1",
                    help="Where to write predictions")
    ap.add_argument("--eval-csv", default="evaluation_qwen_m1.csv",
                    help="Output CSV path")
    ap.add_argument("--hf-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--hf-device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--transcription-dir", default=None,
                    help="Folder containing transcription/<ID>.txt (no line breaks). "
                         "Defaults to <data-dir>/transcription")
    args = ap.parse_args()

    # Instantiate backend
    line_breaker = QwenLineBreaker(QwenCfg(
        model_id=args.hf_model,
        device=args.hf_device,
        max_new_tokens=args.max_new_tokens,
    ))

    gt_dir = os.path.join(args.data_dir, "gt")
    images_root = os.path.join(args.data_dir, "images")
    transcription_dir = args.transcription_dir or os.path.join(args.data_dir, "transcription")

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    if not gt_files:
        print(f"No ground-truth files found in {gt_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    n = 0
    sum_w = sum_c = sum_wn = sum_cn = 0.0
    sum_la = sum_lan = sum_rla = sum_rlan = 0.0

    for gt_path in gt_files:
        sample_id = os.path.splitext(os.path.basename(gt_path))[0]

        # Image(s)
        img_paths = find_images_for_id(images_root, sample_id)
        if not img_paths:
            print(f"[WARN] No images for {sample_id}; skipping.", file=sys.stderr)
            continue

        # Transcription (correct text, no line breaks)
        transcription_path = os.path.join(transcription_dir, f"{sample_id}.txt")
        if not os.path.exists(transcription_path):
            print(f"[WARN] No transcription for {sample_id} in {transcription_dir}; skipping.", file=sys.stderr)
            continue
        transcription = read_text(transcription_path)

        # Ask LLM to only infer line breaks (multi-page)
        try:
            pred = line_breaker.infer_line_breaks(img_paths, transcription)
        except Exception as e:
            print(f"[ERR] Failure for {sample_id}: {e}", file=sys.stderr)
            continue

        write_text(os.path.join(args.out_dir, f"{sample_id}.txt"), pred)

        # ----- Evaluation -----
        gt = read_text(gt_path)

        # token-level metrics
        w  = wer(gt, pred)
        c  = cer(gt, pred)
        wn = wer(normalize_whitespace(gt), normalize_whitespace(pred))
        cn = cer(normalize_whitespace(gt), normalize_whitespace(pred))

        # line-level metrics (Bullinger-style analogue)
        la  = line_accuracy(gt, pred)
        lan = line_accuracy_norm(gt, pred)
        rla = reverse_line_accuracy(gt, pred)
        rlan = reverse_line_accuracy_norm(gt, pred)

        rows.append([
            sample_id,
            len(gt),
            len(pred),
            w, c, wn, cn,
            la, lan,
            rla, rlan,
        ])

        sum_w  += w
        sum_c  += c
        sum_wn += wn
        sum_cn += cn
        sum_la += la
        sum_lan += lan
        sum_rla += rla
        sum_rlan += rlan
        n += 1

        print(
            f"[OK] {sample_id}: "
            f"WER={w:.3f} CER={c:.3f} "
            f"(norm WER={wn:.3f} CER={cn:.3f}) "
            f"LineAcc={la:.3f} LineAcc_norm={lan:.3f} "
            f"RevLineAcc={rla:.3f} RevLineAcc_norm={rlan:.3f}"
        )

    # ----- Write CSV (+ macro average) -----
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
        if n > 0:
            wtr.writerow([])
            wtr.writerow([
                "macro_avg",
                "",
                "",
                sum_w  / n,
                sum_c  / n,
                sum_wn / n,
                sum_cn / n,
                sum_la / n,
                sum_lan / n,
                sum_rla / n,
                sum_rlan / n,
            ])

    print(f"\nWrote {args.eval_csv} with {n} samples.")

if __name__ == "__main__":
    main()
