import importlib
import inspect
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

import ltr.admin.settings as ws_settings


class TRTTrunkBackbone(nn.Module):
    """
    Wraps a torch2trt TRTModule trunk that returns:
      (layer1, layer2, layer3, layer4)

    Exposes a PyTracking/DiMP-compatible API:
      backbone(x, layers=[...]) -> dict
      backbone(x)               -> cls_layer tensor
    """

    def __init__(self, trt_module: nn.Module, layer_names, cls_layer="layer3"):
        super().__init__()
        self.trt = trt_module
        self.layer_names = list(layer_names)
        self.cls_layer = cls_layer

        if self.cls_layer not in self.layer_names:
            raise ValueError(
                f"cls_layer={self.cls_layer!r} not in layer_names={self.layer_names}"
            )

    def forward(self, x, layers=None, output_layers=None):
        if layers is None and output_layers is not None:
            layers = output_layers

        feats = self.trt(x)

        if torch.is_tensor(feats):
            feats = (feats,)

        feat_dict = OrderedDict(zip(self.layer_names, feats))

        if layers is not None:
            missing_layers = [ln for ln in layers if ln not in feat_dict]
            if missing_layers:
                raise KeyError(
                    f"Requested layers {missing_layers} not available. "
                    f"Available: {list(feat_dict.keys())}"
                )
            return OrderedDict((ln, feat_dict[ln]) for ln in layers)

        return feat_dict[self.cls_layer]


class TRTBatchRouter(nn.Module):
    """
    Routes to static TensorRT engines by batch size.

    Preferred:
      B=1  -> B1 engine
      B=13 -> B13 engine

    Fallback:
      unsupported batch -> chunk into B=1 calls and concatenate outputs
    """

    def __init__(self, b1_backbone: nn.Module, b13_backbone: nn.Module = None):
        super().__init__()
        self.b1 = b1_backbone
        self.b13 = b13_backbone

    def _concat_outputs(self, outputs):
        first = outputs[0]

        if isinstance(first, dict):
            merged = OrderedDict()
            for key in first.keys():
                merged[key] = torch.cat([out[key] for out in outputs], dim=0)
            return merged

        return torch.cat(outputs, dim=0)

    def forward(self, x, layers=None, output_layers=None):
        batch = int(x.shape[0])

        if batch == 1:
            return self.b1(x, layers=layers, output_layers=output_layers)

        if batch == 13 and self.b13 is not None:
            return self.b13(x, layers=layers, output_layers=output_layers)

        # Safe fallback for B=2, B=5, etc.
        outs = [
            self.b1(x[i : i + 1], layers=layers, output_layers=output_layers)
            for i in range(batch)
        ]
        return self._concat_outputs(outs)


def _vprint(enabled, message):
    if enabled:
        print(message, flush=True)


def _load_trt_state_or_bundle(path, verbose=False):
    """
    Supports:
      A) raw TRTModule.state_dict()
      B) bundle:
         {
           "trt": state_dict,
           "out_layers": [...],
           "shape": [...],
           "precision": "fp16" / "fp32"
         }
    """
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"TRT bundle not found: {path}")

    obj = torch.load(str(path), map_location="cpu")

    if isinstance(obj, dict) and "trt" in obj:
        _vprint(verbose, f"[ltr_loading] Detected TRT bundle: {path}")
        return obj["trt"], obj.get("out_layers", None), obj.get("shape", None), obj.get("precision", None)

    _vprint(verbose, f"[ltr_loading] Detected raw TRTModule state_dict: {path}")
    return obj, None, None, None


def _make_trtmodule_from_state_dict(trt_state_dict, device="cuda"):
    # Importing from torch2trt.trt_module avoids loading converter registry when possible.
    try:
        from torch2trt.trt_module import TRTModule
    except Exception:
        from torch2trt import TRTModule

    module = TRTModule()
    module.load_state_dict(trt_state_dict)
    module.eval().to(torch.device(device))
    return module


def load_trained_network(workspace_dir, network_path, checkpoint=None):
    """OUTDATED. Use load_pretrained instead."""
    checkpoint_dir = os.path.join(workspace_dir, "checkpoints")
    directory = f"{checkpoint_dir}/{network_path}"
    net, _ = load_network(directory, checkpoint)
    return net


def load_pretrained(module, name, checkpoint=None, **kwargs):
    """Load a network trained using the LTR framework."""
    settings = ws_settings.Settings()
    network_dir = os.path.join(settings.env.workspace_dir, "checkpoints", "ltr", module, name)
    return load_network(network_dir=network_dir, checkpoint=checkpoint, **kwargs)


def _resolve_checkpoint_path(network_dir=None, checkpoint=None):
    net_path = Path(network_dir).expanduser() if network_dir is not None else None

    if net_path is not None and net_path.is_file():
        return str(net_path.resolve())

    if checkpoint is None:
        if net_path is None:
            raise ValueError("Either network_dir or checkpoint must be provided.")

        checkpoint_list = sorted(net_path.glob("*.pth.tar"))

        if not checkpoint_list:
            checkpoint_list = sorted(net_path.glob("*.pth"))

        if not checkpoint_list:
            raise FileNotFoundError(f"No checkpoint file found in {net_path}")

        return str(checkpoint_list[-1].resolve())

    if isinstance(checkpoint, int):
        checkpoint_list = sorted(net_path.glob(f"*_ep{checkpoint:04d}.pth.tar"))
        if not checkpoint_list:
            raise FileNotFoundError(f"No matching checkpoint found for epoch {checkpoint}")
        if len(checkpoint_list) > 1:
            raise RuntimeError(f"Multiple matching checkpoints found for epoch {checkpoint}")
        return str(checkpoint_list[0].resolve())

    if isinstance(checkpoint, str):
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute() and net_path is not None:
            checkpoint_path = net_path / checkpoint_path
        return str(checkpoint_path.resolve())

    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def load_network(
    network_dir=None,
    checkpoint=None,
    constructor_fun_name=None,
    constructor_module=None,
    # TRT options
    trt_backbone_path=None,
    trt_backbone_attr="feature_extractor",
    trt_device="cuda",
    trt_layer_names=("layer1", "layer2", "layer3", "layer4"),
    trt_cls_layer="layer3",
    skip_backbone_weights=True,
    checkpoint_map_location="cpu",
    verbose=False,
    **kwargs,
):
    """
    Load an LTR checkpoint and optionally replace a PyTorch backbone with TensorRT.

    trt_backbone_path:
      None:
        normal PyTorch loading
      str:
        single TRT backbone
      dict:
        {1: path_to_B1_bundle}
        or
        {1: path_to_B1_bundle, 13: path_to_B13_bundle}
    """

    checkpoint_path = _resolve_checkpoint_path(network_dir=network_dir, checkpoint=checkpoint)

    _vprint(verbose, f"[ltr_loading] checkpoint_path = {checkpoint_path}")
    _vprint(verbose, f"[ltr_loading] checkpoint_map_location = {checkpoint_map_location}")

    checkpoint_dict = torch_load_legacy(checkpoint_path, map_location=checkpoint_map_location)

    if "constructor" not in checkpoint_dict or checkpoint_dict["constructor"] is None:
        raise RuntimeError("No constructor found in checkpoint.")

    net_constr = checkpoint_dict["constructor"]

    if constructor_fun_name is not None:
        net_constr.fun_name = constructor_fun_name

    if constructor_module is not None:
        net_constr.fun_module = constructor_module

    if net_constr.fun_module.startswith("dlframework."):
        net_constr.fun_module = net_constr.fun_module[len("dlframework.") :]

    net_fun = getattr(importlib.import_module(net_constr.fun_module), net_constr.fun_name)
    net_fun_args = list(inspect.signature(net_fun).parameters.keys())

    for arg, val in kwargs.items():
        if arg in net_fun_args:
            net_constr.kwds[arg] = val
        else:
            _vprint(verbose, f'WARNING: Keyword argument "{arg}" not found when loading network. Ignored.')

    net = net_constr.get()

    if "net" not in checkpoint_dict:
        raise RuntimeError("Checkpoint does not contain key 'net'.")

    state = checkpoint_dict["net"]

    if trt_backbone_path is not None and skip_backbone_weights:
        prefix = trt_backbone_attr + "."
        _vprint(verbose, f"[ltr_loading] Skipping checkpoint weights with prefix: {prefix!r}")
        state = {k: v for k, v in state.items() if not k.startswith(prefix)}

        missing, unexpected = net.load_state_dict(state, strict=False)

        _vprint(verbose, "[ltr_loading] load_state_dict(strict=False) done.")

        if verbose and missing:
            _vprint(verbose, f"[ltr_loading] Missing keys, expected for TRT backbone: {missing[:10]} ...")

        if verbose and unexpected:
            _vprint(verbose, f"[ltr_loading] Unexpected keys: {unexpected[:10]} ...")
    else:
        net.load_state_dict(state)
        _vprint(verbose, "[ltr_loading] load_state_dict(strict=True) done.")

    net.constructor = checkpoint_dict.get("constructor", None)

    if checkpoint_dict.get("net_info", None) is not None:
        net.info = checkpoint_dict["net_info"]

    if trt_backbone_path is not None:
        if not hasattr(net, trt_backbone_attr):
            raise AttributeError(f"Network has no attribute {trt_backbone_attr!r}")

        layer_names = list(trt_layer_names)

        if isinstance(trt_backbone_path, dict):
            if 1 not in trt_backbone_path:
                raise ValueError("trt_backbone_path dict must contain key 1.")

            p1 = trt_backbone_path[1]
            p13 = trt_backbone_path.get(13, None)

            _vprint(verbose, "[ltr_loading] Loading TRT backbone bundles:")
            _vprint(verbose, f"  B1 : {p1}")
            if p13 is not None:
                _vprint(verbose, f"  B13: {p13}")
            else:
                _vprint(verbose, "  B13: not provided; unsupported batches will use B1 chunk fallback")

            sd1, out1, shape1, precision1 = _load_trt_state_or_bundle(p1, verbose=verbose)

            if out1 is not None:
                layer_names = list(out1)

            m1 = _make_trtmodule_from_state_dict(sd1, device=trt_device)
            bb1 = TRTTrunkBackbone(m1, layer_names, cls_layer=trt_cls_layer).eval().to(torch.device(trt_device))

            bb13 = None

            if p13 is not None:
                sd13, out13, shape13, precision13 = _load_trt_state_or_bundle(p13, verbose=verbose)

                if out13 is not None:
                    layer_names_13 = list(out13)
                    if layer_names_13 != layer_names:
                        raise RuntimeError(
                            f"B1 and B13 output layers differ: {layer_names} vs {layer_names_13}"
                        )

                m13 = _make_trtmodule_from_state_dict(sd13, device=trt_device)
                bb13 = TRTTrunkBackbone(m13, layer_names, cls_layer=trt_cls_layer).eval().to(torch.device(trt_device))

            router = TRTBatchRouter(bb1, bb13).eval().to(torch.device(trt_device))
            setattr(net, trt_backbone_attr, router)

            _vprint(
                verbose,
                f"[ltr_loading] Replaced net.{trt_backbone_attr} with TRTBatchRouter on {trt_device}",
            )

        else:
            _vprint(verbose, f"[ltr_loading] Loading single TRT backbone: {trt_backbone_path}")

            sd, out_layers, shape, precision = _load_trt_state_or_bundle(trt_backbone_path, verbose=verbose)

            if out_layers is not None:
                layer_names = list(out_layers)

            trt_mod = _make_trtmodule_from_state_dict(sd, device=trt_device)
            wrapped = TRTTrunkBackbone(trt_mod, layer_names, cls_layer=trt_cls_layer).eval().to(torch.device(trt_device))

            setattr(net, trt_backbone_attr, wrapped)

            _vprint(
                verbose,
                f"[ltr_loading] Replaced net.{trt_backbone_attr} with TRTTrunkBackbone on {trt_device}",
            )

    return net, checkpoint_dict


def load_weights(net, path, strict=True):
    checkpoint_dict = torch.load(path, map_location="cpu", weights_only=False)
    weight_dict = checkpoint_dict["net"]
    net.load_state_dict(weight_dict, strict=strict)
    return net


def torch_load_legacy(path, map_location="cpu"):
    """Load network with legacy dlframework -> ltr compatibility."""
    _setup_legacy_env()
    checkpoint_dict = torch.load(path, map_location=map_location, weights_only=False)
    _cleanup_legacy_env()
    return checkpoint_dict


def _setup_legacy_env():
    importlib.import_module("ltr")
    sys.modules["dlframework"] = sys.modules["ltr"]
    sys.modules["dlframework.common"] = sys.modules["ltr"]

    importlib.import_module("ltr.admin")
    sys.modules["dlframework.common.utils"] = sys.modules["ltr.admin"]

    for module_name in ("model_constructor", "stats", "settings", "local"):
        importlib.import_module("ltr.admin." + module_name)
        sys.modules["dlframework.common.utils." + module_name] = sys.modules[
            "ltr.admin." + module_name
        ]


def _cleanup_legacy_env():
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("dlframework"):
            del sys.modules[module_name]
