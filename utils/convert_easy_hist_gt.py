#!/usr/bin/env python3
"""
convert_easy_hist_gt.py

Convert the "270-01 ..." encoded GT format into:
- gt/<page_id>.txt  (line-level)
- transcription/<page_id>.txt  (single block, no line breaks)

Usage:
  python convert_easy_hist_gt.py raw_gt.txt easy_hist_dataset
"""

import os
import sys
from collections import defaultdict

# ---------- mapping helpers ----------

def decode_piece(piece: str) -> str:
    """
    Decode a single sub-token (between '-'s), e.g. 's_cm', 's_2', 'W', 'a'.
    """
    if not piece:
        return ""

    # special codes
    if piece.startswith("s_"):
        code = piece[2:]

        # punctuation
        if code == "cm":
            return ","
        if code == "pt":
            return "."
        if code == "sq":
            return ";"
        if code == "qt":
            return "'"  # apostrophe
        if code == "qo":
            return ":"  # colon (best guess)

        # dash / hyphen / em dash
        if code == "mi":
            return "-"  # you can change to "—" if you prefer

        # digits
        if code.isdigit():
            return code  # '2' etc.

        # ordinals like 1st, 2nd, 3rd, 4th, 5th, 6th...
        if code.endswith(("st", "nd", "rd", "th")) and code[:-2].isdigit():
            return code  # "1st", "28th", ...

        # special cases
        if code == "s":   # used in 'unles_s-s' -> 'unless'
            return "s"
        if code == "GW":
            return "GW"

        # fallback: ignore or log
        # print(f"[WARN] Unknown code: {piece}", file=sys.stderr)
        return ""

    # normal letter
    return piece


def decode_token(token: str) -> str:
    """
    Decode a single token (between '|'s), e.g. 'L-e-t-t-e-r-s-s_cm'.
    Returns the decoded string, without surrounding spaces.
    """
    parts = token.split("-")
    chars = [decode_piece(p) for p in parts]
    return "".join(chars)


def decode_line(encoded: str) -> str:
    """
    Decode the full part after '270-01 ' into plain text.
    Tokens are separated by '|'. Some tokens like 's_mi' should
    join without spaces (hyphens/dashes).
    """
    tokens = encoded.split("|")

    out = []
    for tok in tokens:
        text = decode_token(tok)
        if not text:
            continue

        if text == "-":  # hyphen/dash, attach to previous
            if not out:
                out.append(text)
            else:
                out[-1] = out[-1] + text
            continue

        # normal token
        if not out:
            out.append(text)
        else:
            out.append(" " + text)

    return "".join(out).strip()


# ---------- main conversion ----------

def main(raw_gt_path: str, out_root: str) -> None:
    # page_id -> list of decoded lines (in order)
    pages = defaultdict(list)

    with open(raw_gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Example: "270-03 o-n-l-y|f-o-r|t-h-e|..."
            try:
                id_part, enc_part = line.split(maxsplit=1)
            except ValueError:
                # line without content, skip
                continue

            page_id = id_part.split("-")[0]  # "270-03" -> "270"
            decoded = decode_line(enc_part)
            pages[page_id].append(decoded)

    gt_dir = os.path.join(out_root, "gt")
    tr_dir = os.path.join(out_root, "transcription")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(tr_dir, exist_ok=True)

    for page_id, lines in sorted(pages.items(), key=lambda kv: kv[0]):
        # write gt/<page_id>.txt
        gt_path = os.path.join(gt_dir, f"{page_id}.txt")
        with open(gt_path, "w", encoding="utf-8") as f_gt:
            for l in lines:
                f_gt.write(l + "\n")

        # write transcription/<page_id>.txt (no line breaks)
        flat = " ".join(l.strip() for l in lines if l.strip())
        tr_path = os.path.join(tr_dir, f"{page_id}.txt")
        with open(tr_path, "w", encoding="utf-8") as f_tr:
            f_tr.write(flat + "\n")

        print(f"[OK] Wrote {gt_path} and {tr_path}")

    print(f"\nDone. Pages converted: {len(pages)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_easy_hist_gt.py raw_gt.txt easy_hist_dataset", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
