"""Database access for the GEMINI multi-hospital data cut.

See ``docs/gemini.md`` for the full workflow: nobody on this team but Amrit
has a login on the GEMINI node, so everything under
:mod:`odyssey.data.gemini` is designed to be pushed as a script, run once on
the node, and reported back through small text/JSON/HTML output -- never
patient-level data, never a model checkpoint.
"""
