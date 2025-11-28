#!/usr/bin/env python3
"""
copy_pages_to_images.py

Copies all page images from:
    <dataset_root>/pages/
to:
    <dataset_root>/images/<page_id>/<page_id>.<ext>

Assumes filenames like:
    270.jpg
    270.png
    271.jpeg
    10317.tif

Usage:
    python copy_pages_to_images.py easy_hist_dataset
"""

import os
import sys
import shutil

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def main(dataset_root: str):
    pages_dir = os.path.join(dataset_root, "pages")
    images_root = os.path.join(dataset_root, "images")

    if not os.path.isdir(pages_dir):
        print(f"ERROR: folder not found: {pages_dir}")
        sys.exit(1)

    os.makedirs(images_root, exist_ok=True)

    files = sorted(os.listdir(pages_dir))
    count = 0

    for fname in files:
        fpath = os.path.join(pages_dir, fname)
        if not os.path.isfile(fpath):
            continue

        base, ext = os.path.splitext(fname)
        if ext.lower() not in VALID_EXTS:
            continue

        # Extract page ID (numbers before the extension)
        # Example: 270.jpg -> page_id="270"
        page_id = "".join(ch for ch in base if ch.isdigit())
        if page_id == "":
            print(f"[WARN] Skipping {fname}: no leading digits found.")
            continue

        # Create target dir: images/<page_id>/
        out_dir = os.path.join(images_root, page_id)
        os.makedirs(out_dir, exist_ok=True)

        # Copy file to e.g. images/270/270.png
        out_path = os.path.join(out_dir, f"{page_id}{ext.lower()}")
        shutil.copyfile(fpath, out_path)

        print(f"[OK] Copied {fname} -> {out_path}")
        count += 1

    print(f"\nDone. {count} images copied into structured folders.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python copy_pages_to_images.py <dataset_root>")
        sys.exit(1)
    main(sys.argv[1])
