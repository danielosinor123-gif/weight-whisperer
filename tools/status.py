import os, time
from pathlib import Path

base = Path(r"C:\Users\OSINOR TRILLIONS\.local\share\opencode\shell\dd241239e01f1002f5e24c6219255778a4309ed2")
rows = []
for f in base.glob("*.out"):
    rows.append((f.stat().st_mtime, f.stat().st_size, f.name))
rows.sort(reverse=True)
now = time.time()
for m, s, n in rows[:6]:
    age = int(now - m)
    print(f"{time.strftime('%H:%M:%S', time.localtime(m))}  {s/1e6:8.2f} MB  age {age}s  {n}")
