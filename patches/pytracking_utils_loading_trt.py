import os
from pathlib import Path

import ltr.admin.loading as ltr_loading
from pytracking.evaluation.environment import env_settings


def _str_to_bool(value, default=False):
    if value is None:
        return default
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def _resolve_network_path(net_path):
    """Resolve PyTracking network path from absolute path or env_settings().network_path."""
    if os.path.isabs(net_path):
        return Path(net_path).expanduser().resolve()

    network_path = env_settings().network_path

    if isinstance(network_path, (list, tuple)):
        for base in network_path:
            candidate = Path(base).expanduser().resolve() / net_path
            if candidate.exists():
                return candidate
        # Fall back to first base for error clarity.
        return Path(network_path[0]).expanduser().resolve() / net_path

    return Path(network_path).expanduser().resolve() / net_path


def _find_trt_bundle(conversion_dir, batch, height=288, width=288):
    """
    Find TRT bundle for a batch size.

    Priority:
      1. Explicit env var DIMP50_TRT_B1 / DIMP50_TRT_B13
      2. Preferred precision from DIMP50_TRT_PRECISION
      3. fp16 if exists
      4. fp32 if exists
    """
    env_name = f"DIMP50_TRT_B{batch}"
    explicit = os.environ.get(env_name)

    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{env_name} points to missing file: {path}")
        return path

    precision = os.environ.get("DIMP50_TRT_PRECISION", "fp16").lower()

    candidates = [
        conversion_dir / f"dimp50_trunk_B{batch}_3x{height}x{width}_{precision}_trt.pth",
        conversion_dir / f"dimp50_trunk_B{batch}_3x{height}x{width}_fp16_trt.pth",
        conversion_dir / f"dimp50_trunk_B{batch}_3x{height}x{width}_fp32_trt.pth",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find DiMP50 TensorRT bundle for "
        f"B={batch}, input=3x{height}x{width} in {conversion_dir}. "
        "Expected one of:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


def _should_use_trt(path_full):
    """
    Auto-enable TRT for runtime checkpoints unless disabled.

    Disable:
      DIMP50_USE_TRT=0
    Force enable:
      DIMP50_USE_TRT=1
    """
    env_value = os.environ.get("DIMP50_USE_TRT")
    if env_value is not None:
        return _str_to_bool(env_value)

    # Auto-enable only for the runtime checkpoint.
    return path_full.name == "dimp50_runtime.pth"


def load_network(net_path, **kwargs):
    """
    Load network for tracking.

    This patch keeps normal PyTracking behavior, but if the requested checkpoint is
    dimp50_runtime.pth, it injects TensorRT backbone bundles through ltr.admin.loading.

    Expected conversion directory:
      /home/jetson/Desktop/pytracking/conversion/

    Expected files:
      dimp50_runtime.pth
      dimp50_trunk_B1_3x288x288_fp16_trt.pth or fp32
      dimp50_trunk_B13_3x288x288_fp16_trt.pth or fp32
    """
    kwargs["backbone_pretrained"] = False

    path_full = _resolve_network_path(net_path)

    if not path_full.exists():
        raise FileNotFoundError(f"Network checkpoint not found: {path_full}")

    use_trt = _should_use_trt(path_full)

    if use_trt:
        height = int(os.environ.get("DIMP50_TRT_H", "288"))
        width = int(os.environ.get("DIMP50_TRT_W", "288"))
        conversion_dir = Path(os.environ.get("DIMP50_CONVERSION_DIR", str(path_full.parent))).expanduser().resolve()

        b1 = _find_trt_bundle(conversion_dir, batch=1, height=height, width=width)

        # B13 is optional if augmentation is disabled, but recommended.
        # If missing, the LTR router can still use B=1 chunk fallback.
        try:
            b13 = _find_trt_bundle(conversion_dir, batch=13, height=height, width=width)
            trt_paths = {1: str(b1), 13: str(b13)}
        except FileNotFoundError:
            trt_paths = {1: str(b1)}

        verbose = _str_to_bool(os.environ.get("DIMP50_TRT_VERBOSE"), default=True)

        net, _ = ltr_loading.load_network(
            network_dir=str(path_full),
            trt_backbone_path=trt_paths,
            trt_backbone_attr="feature_extractor",
            trt_device=os.environ.get("DIMP50_TRT_DEVICE", "cuda"),
            trt_layer_names=("layer1", "layer2", "layer3", "layer4"),
            trt_cls_layer="layer3",
            skip_backbone_weights=True,
            checkpoint_map_location=os.environ.get("DIMP50_CHECKPOINT_MAP_LOCATION", "cpu"),
            verbose=verbose,
            **kwargs,
        )
    else:
        net, _ = ltr_loading.load_network(
            network_dir=str(path_full),
            **kwargs,
        )

    return net
