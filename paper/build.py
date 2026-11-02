#!/usr/bin/env python3
"""Inline the table fragments (\\input at row start breaks tabular) and compile with tectonic (or xelatex/pdflatex if tectonic missing).
usage: python build.py article_en article_it"""
import re, subprocess, sys, shutil
from pathlib import Path
for name in sys.argv[1:]:
    s = Path(name + '.tex').read_text()
    s = re.sub(r'\\input\{(tab_[a-z0-9_]+)\}', lambda m: Path(m.group(1) + '.tex').read_text(), s)
    Path('_' + name + '.tex').write_text(s)
    eng = ['tectonic'] if shutil.which('tectonic') else ['pdflatex', '-interaction=nonstopmode']
    subprocess.run(eng + ['_' + name + '.tex'], check=True)
    Path('_' + name + '.pdf').replace(name + '.pdf'); Path('_' + name + '.tex').unlink()
