"""Versioned, inference-only export with an explicit native tensor layout.

The native file is self-contained; loading it never imports Python or executes
pickled objects.  The companion safetensors and manifest are useful for auditing
and writing renderer-specific upload code, but are not needed by the C++ loader.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import save_file
from torch import nn
from zuko.nn import Linear as ZukoLinear

from .coupling import AxialSymmetricCouplingTransform

if TYPE_CHECKING:
    from .model import PhaseFlow

MAGIC = b"PHFLOW01"
FORMAT_VERSION = 1
ENDIAN_SENTINEL = 0x01020304
HEADER = struct.Struct("<8sIIQQ10I6d")
NETWORK = struct.Struct("<4I")
DENSE = struct.Struct("<IIQQ")
COUPLING = struct.Struct("<II")
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_WIDTH = 65536
MAX_NETWORKS = 4096
MAX_DENSE_LAYERS = 16384
MAX_BINS = 4096


def _linear_layers(module: nn.Module, name: str) -> list[nn.Linear | ZukoLinear]:
    """Reject networks whose executable behavior cannot be represented in v1."""
    children = list(module.children())
    if not children or len(children) % 2 != 1:
        raise ValueError(f"{name} must be Linear, ReLU, ..., Linear")
    linears: list[nn.Linear | ZukoLinear] = []
    for index, child in enumerate(children):
        if index % 2 == 0 and not isinstance(child, (nn.Linear, ZukoLinear)):
            raise ValueError(f"{name}[{index}] must be Linear")
        if index % 2 == 1 and not isinstance(child, nn.ReLU):
            raise ValueError(f"{name}[{index}] must be ReLU")
        if isinstance(child, (nn.Linear, ZukoLinear)):
            if child.bias is None:
                raise ValueError(f"{name}[{index}] must have a bias")
            if child.weight.shape != (
                child.out_features,
                child.in_features,
            ) or child.bias.shape != (child.out_features,):
                raise ValueError(f"{name}[{index}] must be an unstacked dense layer")
            if not 1 <= child.in_features <= MAX_WIDTH:
                raise ValueError(f"{name}[{index}] input width is unsupported")
            if not 1 <= child.out_features <= MAX_WIDTH:
                raise ValueError(f"{name}[{index}] output width is unsupported")
            if linears and linears[-1].out_features != child.in_features:
                raise ValueError(f"{name} layer dimensions do not form a chain")
            linears.append(child)
    return linears


def export_model(model: PhaseFlow, output_directory: str | os.PathLike[str]) -> Path:
    """Export FP32 weights and return the directory containing all three files.

    Files: ``model.pflow``, ``manifest.json``, and ``weights.safetensors``.
    Numerical inference in the C++ reference uses double precision and converts
    the exported FP32 weights at each multiply.  This is a correctness reference,
    not a claim that the native and PyTorch FP32 arithmetic are bitwise equal.
    Unsupported architectures or nonfinite parameters fail before writing files.
    """
    config = model.config
    if config.activation != "relu":
        raise ValueError("Export format v1 supports only ReLU hidden activations")
    if not 0 < config.wavelength_min_nm < config.wavelength_max_nm:
        raise ValueError("Wavelength bounds must be positive and increasing")
    if not 0 < config.g_limit < 1:
        raise ValueError("g_limit must be strictly between zero and one")
    scalar_config = (
        float(config.wavelength_min_nm),
        float(config.wavelength_max_nm),
        float(config.g_limit),
        float(config.min_bin_width),
        float(config.min_bin_height),
        float(config.min_derivative),
    )
    if not all(math.isfinite(value) for value in scalar_config):
        raise ValueError("Configuration contains nonfinite values")
    bins = int(config.num_bins)
    blob_bins = int(config.one_blob_bins)
    if bins != config.num_bins or not 1 <= bins <= MAX_BINS:
        raise ValueError("num_bins is outside the native format limits")
    if blob_bins != config.one_blob_bins or not 0 <= blob_bins <= MAX_BINS:
        raise ValueError("one_blob_bins is outside the native format limits")
    if not 0 < bins * config.min_bin_width < 1:
        raise ValueError("num_bins * min_bin_width must be in (0, 1)")
    if not 0 < bins * config.min_bin_height < 1 or config.min_derivative <= 0:
        raise ValueError("Spline height and derivative minima must be positive")
    if config.include_g_context not in (False, True):
        raise ValueError("include_g_context must be a boolean")
    include_g = int(config.include_g_context)
    encoded_size = 2 + 2 * blob_bins
    context_size = encoded_size + include_g
    if (
        model.encoder.bins != blob_bins
        or model.encoder.wavelength_min_nm != config.wavelength_min_nm
        or model.encoder.wavelength_max_nm != config.wavelength_max_nm
        or model.g_head.g_limit != config.g_limit
    ):
        raise ValueError("The condition encoder or HG head differs from its configuration")
    coupling_modules = list(model.couplings)
    if len(coupling_modules) != config.num_coupling_layers:
        raise ValueError("Coupling count does not match the model configuration")
    if not 0 <= len(coupling_modules) < MAX_NETWORKS:
        raise ValueError("Too many coupling layers for native format v1")
    for coupling in coupling_modules:
        if type(coupling) is not AxialSymmetricCouplingTransform:
            raise ValueError("Export v1 requires PhaseFlow's axial-symmetric RQS coupling")
        for field in ("num_bins", "min_bin_width", "min_bin_height", "min_derivative"):
            if getattr(coupling, field) != getattr(config, field):
                raise ValueError(f"Coupling {field} differs from its configuration")
    modules = [("g", model.g_head.mlp)] + [
        (f"coupling.{index}", coupling.hyper) for index, coupling in enumerate(coupling_modules)
    ]
    networks: list[tuple[int, int, int, int]] = []
    dense_layers: list[tuple[int, int, int, int]] = []
    couplings: list[tuple[int, int]] = []
    tensors: dict[str, torch.Tensor] = {}
    tensor_descriptors: list[dict] = []
    parameter_chunks: list[bytes] = []
    parameter_count = 0
    workspace_width = max(encoded_size, context_size + 1)
    for network_index, (name, module) in enumerate(modules):
        linears = _linear_layers(module, name)
        expected_input = encoded_size if network_index == 0 else context_size + 1
        expected_output = 1 if network_index == 0 else 3 * bins + 1
        if linears[0].in_features != expected_input or linears[-1].out_features != expected_output:
            raise ValueError(f"{name} input/output sizes do not match the model contract")
        first_layer = len(dense_layers)
        for layer_index, linear in enumerate(linears):
            offsets = []
            for kind, parameter in (("weight", linear.weight), ("bias", linear.bias)):
                tensor_name = f"{name}.layer{layer_index}.{kind}"
                tensor = (
                    parameter.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
                )
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"{tensor_name} has nonfinite FP32 parameters")
                offsets.append(parameter_count)
                tensors[tensor_name] = tensor
                tensor_descriptors.append(
                    {
                        "name": tensor_name,
                        "shape": list(tensor.shape),
                        "element_offset": parameter_count,
                        "element_count": tensor.numel(),
                    }
                )
                # numpy emits the requested little-endian bytes on every host.
                parameter_chunks.append(tensor.numpy().astype("<f4", copy=False).tobytes(order="C"))
                parameter_count += tensor.numel()
            dense_layers.append((linear.in_features, linear.out_features, *offsets))
            workspace_width = max(workspace_width, linear.in_features, linear.out_features)
        networks.append((first_layer, len(linears), expected_input, expected_output))
    if len(dense_layers) > MAX_DENSE_LAYERS:
        raise ValueError("Too many dense layers for native format v1")
    for index, coupling in enumerate(coupling_modules):
        mask = coupling.mask.detach().to(device="cpu")
        if mask.dtype != torch.bool or mask.shape != (2,) or int(mask.sum()) != 1:
            raise ValueError(f"Coupling {index} must retain exactly one of two coordinates")
        retained_index = int(torch.nonzero(mask, as_tuple=False)[0, 0])
        if retained_index != (1 if index % 2 == 0 else 0):
            raise ValueError("Coupling masks must match PhaseFlow's alternating coordinate order")
        couplings.append((index + 1, retained_index))
    parameter_bytes = b"".join(parameter_chunks)
    total_bytes = (
        HEADER.size
        + NETWORK.size * len(networks)
        + DENSE.size * len(dense_layers)
        + COUPLING.size * len(couplings)
        + len(parameter_bytes)
    )
    if total_bytes > MAX_FILE_BYTES:
        raise ValueError("Export exceeds the native loader's 512 MiB file limit")
    header = HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        ENDIAN_SENTINEL,
        total_bytes,
        parameter_count,
        len(dense_layers),
        len(networks),
        len(couplings),
        blob_bins,
        bins,
        include_g,
        1,
        workspace_width,
        0,
        0,
        *scalar_config,
    )
    binary = b"".join(
        [
            header,
            *(NETWORK.pack(*item) for item in networks),
            *(DENSE.pack(*item) for item in dense_layers),
            *(COUPLING.pack(*item) for item in couplings),
            parameter_bytes,
        ]
    )
    manifest = {
        "format": "phaseflow",
        "version": FORMAT_VERSION,
        "native_file": "model.pflow",
        "native_sha256": hashlib.sha256(binary).hexdigest(),
        "native_bytes": len(binary),
        "weights_file": "weights.safetensors",
        "weight_dtype": "float32",
        "weight_byte_order": "little",
        "weight_matrix_layout": "row_major_out_features_in_features",
        "native_reference_arithmetic": "float64",
        "rounded_pole_sample_policy": "invalid; no clamping, density substitution, or retry",
        "cuda_validation": "not_compiled_or_executed_in_this_project_environment",
        "condition_order": ["wavelength_nm", "incident_cosine"],
        "incident_cosine_definition": "particle_axis dot incident_propagation_direction",
        "normalized_conditions": [
            "(wavelength_nm-lambda_min)/(lambda_max-lambda_min)",
            "(incident_cosine+1)/2",
        ],
        "encoding": {
            "type": "raw_then_integrated_gaussian_one_blob",
            "order": ["s_lambda", "s_eta", "lambda_bins", "eta_bins"],
            "sigma": None if blob_bins == 0 else 1 / blob_bins,
            "renormalize_boundary_mass": False,
        },
        "g_head": "g_limit * tanh(MLP(encoded_conditions))",
        "conditioner_input_order": ["retained_coordinate", "encoded_conditions"]
        + (["raw_g"] if include_g else []),
        "spline_logits_order": ["widths[K]", "heights[K]", "derivatives[K+1]"],
        "axial_incidence_symmetry": {
            "enabled": True,
            "strength": "a=sqrt((1-incident_cosine)*(1+incident_cosine))",
            "radial_conditioner_retained_input": "0.5+a*(folded_azimuth-0.5)",
            "azimuth_mass_blend_before_normalization": "a*constrained_mass+(1-a)/num_bins",
            "azimuth_derivative_blend": "a*constrained_derivative+(1-a)",
        },
        "flow_direction": "data_to_base; sampling traverses inverse layers in reverse order",
        "density_measure": "solid_angle_sr^-1",
        "coordinates": {
            "direction": "local_z_incident_propagation_x_projected_particle_axis",
            "square": ["HG_CDF(direction.z; g)", "abs(atan2(y,x))/pi"],
            "uniform_domain": "(0,1) x [0,1)",
            "mirror_sign": "negative if u[1]<0.5, positive otherwise",
            "base_folded_azimuth": "2*u[1] mod 1",
            "pole_azimuth": "canonical zero for evaluation",
        },
        "config": config.to_dict(),
        "workspace": {
            "max_network_width": workspace_width,
            "scalar_count_per_query": context_size + 2 * workspace_width,
        },
        "networks": [
            dict(zip(("first_layer", "num_layers", "input_size", "output_size"), net))
            for net in networks
        ],
        "dense_layers": [
            dict(zip(("input_size", "output_size", "weight_offset", "bias_offset"), layer))
            for layer in dense_layers
        ],
        "couplings": [
            {"network_index": net, "retained_index": retained} for net, retained in couplings
        ],
        "tensors": tensor_descriptors,
    }
    directory = Path(output_directory).expanduser().resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Stage every artifact after validation. Individual replacements are atomic;
    # consumers should open the native file, which is independently self-contained.
    with tempfile.TemporaryDirectory(prefix=".phaseflow-export-", dir=directory.parent) as stage:
        staging = Path(stage)
        (staging / "model.pflow").write_bytes(binary)
        save_file(
            tensors,
            str(staging / "weights.safetensors"),
            metadata={
                "format": "phaseflow",
                "version": str(FORMAT_VERSION),
                "dtype": "float32",
            },
        )
        manifest["weights_sha256"] = hashlib.sha256(
            (staging / "weights.safetensors").read_bytes()
        ).hexdigest()
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        directory.mkdir(parents=True, exist_ok=True)
        for filename in ("weights.safetensors", "manifest.json", "model.pflow"):
            os.replace(staging / filename, directory / filename)
    return directory
