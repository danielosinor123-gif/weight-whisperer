import os, shutil, sys
from pathlib import Path

C = "C:" + os.sep

def free():
    return shutil.disk_usage(C).free / 1e9

mode = sys.argv[1] if len(sys.argv) > 1 else "scan"

if mode == "scan":
    rows = []
    for e in os.scandir(C):
        try:
            if e.is_dir():
                total = 0
                for e2 in os.scandir(e.path):
                    try:
                        if e2.is_file():
                            total += e2.stat().st_size
                    except OSError:
                        pass
                rows.append((total, e.name + "/"))
            else:
                rows.append((e.stat().st_size, e.name))
        except OSError:
            pass
    rows.sort(reverse=True)
    for sz, name in rows[:15]:
        print(f"{sz/1e9:7.2f} GB  {name}")
    print(f"free now: {free():.2f} GB")

elif mode == "clean":
    freed = 0.0
    hf = Path(os.path.expanduser("~/.cache/huggingface"))
    for root, dirs, files in os.walk(hf):
        for f in files:
            if f.endswith(".incomplete"):
                fp = os.path.join(root, f)
                try:
                    freed += os.path.getsize(fp)
                    os.remove(fp)
                except OSError:
                    pass
    pip = Path(os.path.expanduser("~/.cache/pip"))
    if pip.exists():
        for r, d, fs in os.walk(pip):
            for f in fs:
                try:
                    freed += os.path.getsize(os.path.join(r, f))
                except OSError:
                    pass
        shutil.rmtree(pip, ignore_errors=True)
    oc = Path(os.path.expanduser("~/.local/share/opencode/shell"))
    if oc.exists():
        for f in oc.glob("*.out"):
            try:
                if f.stat().st_size > 5e6:
                    freed += f.stat().st_size
                    f.unlink()
            except OSError:
                pass
    print(f"freed {freed/1e9:.2f} GB; free now: {free():.2f} GB")
