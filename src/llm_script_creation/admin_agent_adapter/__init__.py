"""Read-only adapter to the real Uwazi admin agent's extraction machinery.

The benchmark's main process must not import ``uwazi_admin_agent`` (that would
drag pydantic-ai / rapidfuzz / dateutil into our environment). The package is
split accordingly:

* main-process safe (stdlib + our package only): :mod:`prompt`, :mod:`evaluate`,
  :mod:`sampling` and the request/response plumbing in :mod:`cli`.
* sidecar-only (imports ``uwazi_admin_agent`` / ``uwazi_agent``): :mod:`records`,
  :mod:`ports`, :mod:`runner`, :mod:`sandbox`. They run inside
  ``scripts/admin_agent_sidecar.py`` with the admin agent's own venv.

This module intentionally exports nothing: importing the package must stay safe
in the benchmark environment.
"""

from __future__ import annotations

__all__: list[str] = []
