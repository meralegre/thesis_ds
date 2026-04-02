from pathlib import Path
import re

src_root = Path("../data/meertens_clean")
dst_root = Path("../data/meertens_minimal2")

dur_re = re.compile(r'^(\d+\.*)([a-gA-Gr])')  # must start with duration then pitch/rest

created = 0
skipped = 0

for f in src_root.rglob("*.krn"):
    text = f.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    
    # Extract header info from kern spine
    key_sig = None
    meter = None
    tempo = None
    mode = None
    
    for line in lines:
        tok = line.strip().split("\t")[0].strip()
        if tok.startswith("*k["):
            key_sig = tok
        if tok.startswith("*M") and not tok.startswith("*MM"):
            meter = tok
        if tok.startswith("*MM"):
            tempo = tok
        if re.match(r'^\*[a-gA-G]', tok):
            mode = tok
    
    # Extract only valid note tokens from kern spine
    notes = []
    for line in lines:
        tok = line.strip().split("\t")[0].strip()
        
        # Keep barlines
        if tok.startswith("="):
            # Clean barline — keep only = and digits
            clean_bar = "".join(c for c in tok if c in "0123456789=:|!")
            if clean_bar:
                notes.append(clean_bar)
            continue
        
        # Skip non-note lines
        if not tok or tok.startswith(("*", "!", ".")):
            continue
        
        # Skip grace notes entirely
        if "q" in tok or "Q" in tok:
            continue
        
        # Strip to essentials: duration + pitch + accidentals + ties
        clean = ""
        for ch in tok:
            if ch in "0123456789.abcdefgABCDEFG#-nr[]{}()_":
                clean += ch
        
        if not clean:
            continue
        
        # Strip leading tie/phrase markers to check structure
        inner = clean.lstrip("[{(")
        
        # MUST start with a digit (duration) or be a rest with duration
        if not inner or not inner[0].isdigit():
            # Skip — this would give IDyOM a NIL DUR
            continue
        
        # Verify it actually has a pitch or rest after the duration
        if dur_re.match(inner) or inner.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
            notes.append(clean)
    
    # Must have at least 2 real notes
    real_notes = [n for n in notes if not n.startswith("=")]
    if len(real_notes) < 2:
        skipped += 1
        continue
    
    # Build minimal kern file
    out_lines = ["**kern", "*clefG2"]
    if meter:
        out_lines.append(meter)
    else:
        out_lines.append("*M4/4")
    if tempo:
        out_lines.append(tempo)
    if key_sig:
        out_lines.append(key_sig)
    if mode:
        out_lines.append(mode)
    
    out_lines.extend(notes)
    out_lines.append("*-")
    
    rel = f.relative_to(src_root)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_lines), encoding="utf-8")
    created += 1

print(f"Files created: {created}")
print(f"Skipped (too short): {skipped}")
print(f"Output: {dst_root}")
