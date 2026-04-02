from pathlib import Path
from collections import Counter
import re

dur_re = re.compile(r'^(\d+\.*)') 

all_durations = Counter()
files_per_dur = {}

for f in Path("../data/meertens_unique").rglob("*.krn"):
    text = f.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("*", "!", "=", ".")):
            continue
        for tok in line.split("\t"):
            tok = tok.strip()
            if not tok:
                continue
            m = dur_re.match(tok)
            if m:
                dur = m.group(1)
                all_durations[dur] += 1
                if dur not in files_per_dur:
                    files_per_dur[dur] = []
                if f.name not in files_per_dur[dur]:
                    files_per_dur[dur].append(f.name)

print("All kern durations found:")
for dur, count in all_durations.most_common():
    ticks = None
    try:
        base = int(dur.rstrip("."))
        dots = dur.count(".")
        t = 96 / base
        for d in range(dots):
            t += 96 / base / (2 ** (d + 1))
        ticks = t
    except:
        pass
    is_int = ticks is not None and ticks == int(ticks)
    flag = "" if is_int else " ← NON-INTEGER TICKS"
    n_files = len(files_per_dur.get(dur, []))
    print(f"  {dur:>6s}: {count:>8,} occurrences in {n_files:>5} files  (={ticks} ticks){flag}")
