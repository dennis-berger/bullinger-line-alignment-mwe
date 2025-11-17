# metrics.py
from typing import List

def _lev(a: List[str], b: List[str]) -> int:
    """Levenshtein distance on sequences a, b."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = tmp
    return dp[m]

def wer(ref: str, hyp: str) -> float:
    """Word Error Rate."""
    rt, ht = ref.strip().split(), hyp.strip().split()
    if not rt:
        return 0.0 if not ht else 1.0
    return _lev(rt, ht) / max(1, len(rt))

def cer(ref: str, hyp: str) -> float:
    """Character Error Rate."""
    rc, hc = list(ref.strip()), list(hyp.strip())
    if not rc:
        return 0.0 if not hc else 1.0
    return _lev(rc, hc) / max(1, len(rc))

def normalize_whitespace(s: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return " ".join(s.split())

# -------- Line-level accuracy (Bullinger-style) --------

def _split_lines(s: str) -> List[str]:
    # keep line order, strip trailing/leading whitespace per line
    if not s.strip():
        return []
    return [line.strip() for line in s.strip().splitlines()]

def line_accuracy(ref: str, hyp: str) -> float:
    """
    Line-level accuracy:
    fraction of lines that match exactly (per index).

    - Compare line i of ref with line i of hyp.
    - If ref or hyp has fewer lines, missing lines are treated as empty.
    - Exact string match required for a line to be counted as correct.
    """
    ref_lines = _split_lines(ref)
    hyp_lines = _split_lines(hyp)

    if not ref_lines and not hyp_lines:
        return 1.0
    if not ref_lines and hyp_lines:
        return 0.0

    n = max(len(ref_lines), len(hyp_lines))
    correct = 0
    for i in range(n):
        r = ref_lines[i] if i < len(ref_lines) else ""
        h = hyp_lines[i] if i < len(hyp_lines) else ""
        if r == h:
            correct += 1
    return correct / n

def line_accuracy_norm(ref: str, hyp: str) -> float:
    """
    Normalized line-level accuracy:
    same as line_accuracy, but normalize whitespace inside each line first.
    """
    def norm_line(line: str) -> str:
        return normalize_whitespace(line)

    ref_lines = [norm_line(l) for l in _split_lines(ref)]
    hyp_lines = [norm_line(l) for l in _split_lines(hyp)]

    if not ref_lines and not hyp_lines:
        return 1.0
    if not ref_lines and hyp_lines:
        return 0.0

    n = max(len(ref_lines), len(hyp_lines))
    correct = 0
    for i in range(n):
        r = ref_lines[i] if i < len(ref_lines) else ""
        h = hyp_lines[i] if i < len(hyp_lines) else ""
        if r == h:
            correct += 1
    return correct / n
