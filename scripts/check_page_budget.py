"""Count body lines that spill past the ML4H page-8 limit.

ML4H allows 8 pages excluding references and appendices, so the question
is where the BODY stops, not where the bibliography starts. Two earlier
checks got this wrong in ways that both read as "fine":

* grepping the PDF text for "References" finds where the bibliography
  begins, which is on page 9 by design and says nothing about the body;
* ``line[:8].strip().isdigit()`` never matches, because that slice
  includes the first characters of the line's text, so it reported one
  spilling line while twelve were spilling.

The paper runs ``lineno``, so every body line carries a left-margin
number and bibliography text does not. Counting numbered lines on page 9
before the References heading is therefore the measure that matches the
rule. Counting non-empty lines instead sweeps in the right column's
bibliography and over-counts.

Usage::

    uv run python scripts/check_page_budget.py paper/ml4h/main.pdf

Exits non-zero when the body spills, so it can gate a build.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


BODY_LINE = re.compile(r"^\s{0,6}\d{1,4}\s{2,}\S")


def spilling_lines(pdf: Path, limit_page: int = 8) -> list[str]:
    """Numbered body lines appearing after ``limit_page``."""
    text = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    pages = text.split("\f")
    if len(pages) <= limit_page:
        return []
    page = pages[limit_page].split("\n")
    stop = next((i for i, ln in enumerate(page) if "References" in ln), len(page))
    return [ln for ln in page[:stop] if BODY_LINE.match(ln)]


def main() -> int:
    """Report the spill and return 1 when the body overruns."""
    pdf = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/ml4h/main.pdf")
    if not pdf.exists():
        print(f"no such pdf: {pdf}")
        return 2
    spill = spilling_lines(pdf)
    print(f"numbered body lines past page 8: {len(spill)}")
    for line in spill:
        print("   ", line[:78].rstrip())
    return 1 if spill else 0


if __name__ == "__main__":
    raise SystemExit(main())
