# GEMINI

No `GEMINI.yaml` here, unlike `eICU.yaml` in this same directory --
GEMINI's extraction is SQL-based, not MESSY-based.

GEMINI has no file distribution at all (the only access is a live SQL
connection to the GEMINI node), and `meds-extract-run`'s MESSY tooling assumes
a file-based `input_dir`. Making it fit would mean dumping GEMINI's tables
to files first: roughly doubling the enclave's storage footprint for no
functional benefit, and needing to validate the whole `meds-extract-run`
toolchain inside the closed enclave, untested and unnecessary. Extraction
queries GEMINI directly and writes MEDS parquet shards straight out instead
-- see `scripts/gemini/extract_meds.py` (the actual spec, one function per
source table) and `docs/gemini_extraction.md` (the design doc, open
questions, and the full "why no MESSY spec" reasoning).

The output is standard MEDS parquet either way -- downstream is
source-agnostic.
