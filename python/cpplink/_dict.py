# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""A generic ``to_dict`` over the bound report objects.

Every report is a C++ struct bound with one read-only property per field, so a
dictionary is had by walking the class for those properties rather than by a
second list of field names that would have to be kept in sync by hand. ``text``
is left out, being the printed form of the same fields.
"""

from __future__ import annotations

from typing import Any

_SKIP = frozenset({"text"})


def _properties(cls: type) -> list[str]:
    names: list[str] = []
    for klass in cls.__mro__:
        for name, value in vars(klass).items():
            if name.startswith("_") or name in _SKIP or name in names:
                continue
            # pybind11 exposes `def_readonly` and `def_property_readonly` as
            # descriptors with a getter and no setter.
            if isinstance(value, property) or type(value).__name__ in {
                "getset_descriptor",
                "member_descriptor",
            }:
                names.append(name)
    return names


def _is_bound(obj: Any) -> bool:
    module = type(obj).__module__ or ""
    return module.startswith("cpplink._cpplink") or module == "cpplink"


def to_dict(obj: Any) -> Any:
    """The bound object as plain Python: dicts, lists, numbers and strings.

    Bound objects become dicts of their read-only properties, lists and tuples
    recurse, numpy arrays become lists, and everything else is returned as is.
    """
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj
    if isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    if hasattr(obj, "tolist") and hasattr(obj, "dtype"):
        return obj.tolist()
    if _is_bound(obj):
        return {name: to_dict(getattr(obj, name)) for name in _properties(type(obj))}
    return obj
