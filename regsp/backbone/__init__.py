"""Vendored SafeDiff-VLA backbone code (Apache-2.0, see NOTICE).

Importing this package registers the ``safediff_vla`` policy type with LeRobot's config registry so that
``SafeDiffVLAStagePlanV2Policy.from_pretrained`` can read the backbone checkpoint's ``config.json``.

ReGSP must run against *upstream* ``lerobot``. If the SafeDiff research fork of LeRobot (which registers the same
type name) is installed instead, registration would collide; this is reported with an explicit message.
"""
from __future__ import annotations

try:
    from .configuration_safediff_vla import SafeDiffVLAConfig  # noqa: F401  (registers "safediff_vla")
except ValueError as e:  # pragma: no cover - environment misconfiguration
    raise ImportError(
        "the 'safediff_vla' policy type is already registered, most likely by the SafeDiff research fork of "
        "LeRobot. ReGSP vendors that code and needs upstream `lerobot` (see docs/INSTALL.md)."
    ) from e
