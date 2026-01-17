"""DedeuceRL: A Modular Framework for Active System Identification Benchmarks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("dedeucerl")
except PackageNotFoundError:  # pragma: no cover
    # Source checkout without an installed distribution.
    __version__ = "0.0.0"

from dedeucerl.core.env import HiddenSystemEnv
from dedeucerl.core.types import ProbeResult, SubmitResult, EpisodeState
from dedeucerl.core.config import SkinConfig
from dedeucerl.core.rubric import make_rubric, reward_identification

__all__ = [
    "__version__",
    "HiddenSystemEnv",
    "ProbeResult",
    "SubmitResult",
    "EpisodeState",
    "SkinConfig",
    "make_rubric",
    "reward_identification",
]
