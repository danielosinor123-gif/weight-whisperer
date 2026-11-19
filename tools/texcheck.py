import re
import sys
from pathlib import Path

p = Path(sys.argv[1])
t = p.read_text(encoding="utf-8")
depth = 0
bad = []
for n, line in enumerate(t.splitlines(), 1):
    # LaTeX comment: a % not preceded by a backslash
    lc = re.sub(r"(?<!\\)%.*", "", line)
    for i, ch in enumerate(lc):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                bad.append((n, "extra }", line))
                depth = 0
    if depth > 0 and n in (132, 133, 154, 155):
        print("line", n, "end-depth", depth, ":", line[:100])
print("final depth:", depth)
for b in bad[:5]:
    print("BAD", b)
# env balance
s = "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in t.splitlines())
begins = re.findall(r"\\begin\{(\w+)\}", s)
ends = re.findall(r"\\end\{(\w+)\}", s)
print("envs balanced:", sorted(begins) == sorted(ends))
# citations vs bib entries
cites = set()
for m in re.findall(r"\\cite\{([^}]+)\}", s):
    cites.update(c.strip() for c in m.split(","))
bib = Path(str(p).replace("main.tex", "refs.bib")).read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
print("cited:", sorted(cites))
print("in bib:", sorted(keys))
print("missing from bib:", sorted(cites - keys))
print("unused in bib:", sorted(keys - cites))
