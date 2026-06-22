"""
Scan a kern corpus and list every file whose **... header line declares
more than one spine, grouping by the exact spine signature so it's easy
to see whether the "polyphony" is **kern+**text (lyrics) or **kern+**kern
(actual multi-voice).

Usage:
    python list_polyphonic.py <corpus_root> [output.csv]

Without an output path it just prints to stdout.
"""
import sys
from pathlib import Path
from collections import Counter, defaultdict


def first_exinterp_line(path: Path) -> str | None:
    """Return the first line that starts with '**' (the spine header)."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n").rstrip("\r")
                if line.startswith("**"):
                    return line
    except Exception:
        return None
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python list_polyphonic.py <corpus_root> [output.csv]")
        sys.exit(1)

    root = Path(sys.argv[1])
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    rows = []  # (relpath, header, n_spines)
    sig_counter = Counter()
    sig_examples = defaultdict(list)

    for path in sorted(root.rglob("*.krn")):
        header = first_exinterp_line(path)
        if header is None:
            continue
        # split on tab; spines are tab-separated
        spines = header.split("\t")
        n = len(spines)
        if n <= 1:
            continue
        sig = " ".join(spines)
        rel = path.relative_to(root)
        rows.append((str(rel), sig, n))
        sig_counter[sig] += 1
        if len(sig_examples[sig]) < 3:
            sig_examples[sig].append(str(rel))

    print(f"Scanned: {root}")
    print(f"Files with >1 spine: {len(rows)}\n")
    print("Header signatures found (count, signature):")
    for sig, c in sig_counter.most_common():
        print(f"  {c:>5}  {sig}")
        for ex in sig_examples[sig]:
            print(f"         e.g. {ex}")

    if out_csv:
        import csv
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["relpath", "header", "n_spines"])
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
