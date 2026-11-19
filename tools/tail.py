import re, sys
from pathlib import Path

base = Path(r"C:\Users\OSINOR TRILLIONS\.local\share\opencode\shell\dd241239e01f1002f5e24c6219255778a4309ed2")
sid = sys.argv[1]
p = base / (sid + ".out")
if not p.exists():
    print("no output file yet")
    sys.exit(0)
t = p.read_text(encoding="utf-8", errors="replace")
clean = re.sub(r"[^\x20-\x7e\n]", "", t)
lines = [l.strip() for l in clean.splitlines()
         if l.strip() and "it/s" not in l and "it]" not in l and "s/it" not in l]
print("\n".join(lines[-15:]) if lines else f"(only progress bars, {len(t)} bytes)")
