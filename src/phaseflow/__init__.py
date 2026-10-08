"""Conditional spherical phase densities, measured per unit solid angle."""

__version__ = "0.3.0"


def __getattr__(name):
    if name in {"SphereFlowConfig", "SingleConditionSphereFlow"}:
        from .sphere_model import SingleConditionSphereFlow, SphereFlowConfig

        return {
            "SphereFlowConfig": SphereFlowConfig,
            "SingleConditionSphereFlow": SingleConditionSphereFlow,
        }[name]
    if name == "RainbowReference":
        from .rainbow import RainbowReference

        return RainbowReference
    if name in {"ModelConfig", "PhaseFlow"}:
        from .model import ModelConfig, PhaseFlow

        return {"ModelConfig": ModelConfig, "PhaseFlow": PhaseFlow}[name]
    raise AttributeError(name)


__all__ = [
    "ModelConfig",
    "PhaseFlow",
    "SphereFlowConfig",
    "SingleConditionSphereFlow",
    "RainbowReference",
    "__version__",
]
