"""SOC Triage Environment package exports."""

# Guard the client import \u2013 it requires `requests` which may not be present,
# and also uses absolute `soc_triage_env.*` imports that fail when the package
# is not pip-installed (e.g. in the validator's raw-file workspace).
try:
    from soc_triage_env.client import SOCTriageEnvClient
except Exception:  # pragma: no cover
    SOCTriageEnvClient = None  # type: ignore[assignment,misc]

from soc_triage_env.models import TriageAction, TriageObservation, TriageReward, TriageState

__all__ = [
    "SOCTriageEnvClient",
    "TriageAction",
    "TriageObservation",
    "TriageReward",
    "TriageState",
]
