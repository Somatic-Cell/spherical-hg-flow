"""Conditional spherical phase densities, measured per unit solid angle."""

__version__ = "0.1.0"


def __getattr__(name):
    if name in {"ModelConfig", "PhaseFlow"}:
        from .model import ModelConfig, PhaseFlow

        return {"ModelConfig": ModelConfig, "PhaseFlow": PhaseFlow}[name]
    raise AttributeError(name)


__all__ = ["ModelConfig", "PhaseFlow", "__version__"]
