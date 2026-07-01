"""
python conversion/create_runtime_ckpt.py \
  --in /home/jetson/Desktop/pytracking/pytracking/pytracking/networks/dimp50.pth \
  --out /home/jetson/Desktop/pytracking/conversion/dimp50_runtime.pth \
  --drop_prefix feature_extractor. \
  --strip_optimizer
"""

#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import tensorrt as trt
from torch2trt import torch2trt


OUT_LAYERS = ("layer1", "layer2", "layer3", "layer4")


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


class ResNetTrunk(nn.Module):
    """
    Runs DiMP ResNet backbone up to layer4 and returns:
        layer1, layer2, layer3, layer4

    For DiMP50 + 288x288 input, expected shapes are approximately:
        layer1: [B, 256, 72, 72]
        layer2: [B, 512, 36, 36]
        layer3: [B, 1024, 18, 18]
        layer4: [B, 2048, 9, 9]
    """

    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return f1, f2, f3, f4


def save_bundle(path: Path, trt_state_dict, out_layers, shape, precision: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "trt": trt_state_dict,
            "out_layers": list(out_layers),
            "shape": list(shape),
            "precision": precision,
        },
        str(path),
    )

    log(f"Saved bundle: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DiMP50 ResNet50 backbone trunk to TensorRT using torch2trt."
    )

    parser.add_argument(
        "--pytracking-root",
        default="/home/jetson/Desktop/pytracking/pytracking",
        help="Path to the actual PyTracking repository folder containing ltr/ and pytracking/.",
    )

    parser.add_argument(
        "--checkpoint",
        default="/home/jetson/Desktop/pytracking/pytracking/pytracking/networks/dimp50.pth",
        help="Path to dimp50.pth checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        default="/home/jetson/Desktop/pytracking/conversion",
        help="Where to save TensorRT backbone bundles.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=288,
        help="Input height. Must match params.image_sample_size.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=288,
        help="Input width. Must match params.image_sample_size.",
    )

    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 13],
        help="Batch sizes to convert. DiMP commonly needs B=1 and B=13.",
    )

    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="TensorRT precision. fp16 is recommended for Jetson speed.",
    )

    parser.add_argument(
        "--workspace",
        type=int,
        default=1 << 30,
        help="TensorRT max workspace size in bytes.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pytracking_root = Path(args.pytracking_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not pytracking_root.exists():
        raise FileNotFoundError(f"PyTracking root does not exist: {pytracking_root}")

    if not (pytracking_root / "ltr").exists():
        raise FileNotFoundError(
            f"Invalid PyTracking root: {pytracking_root}. Expected ltr/ inside it."
        )

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    sys.path.insert(0, str(pytracking_root))

    import ltr.admin.loading as ltr_loading

    torch.backends.cudnn.benchmark = True
    device = "cuda"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. TensorRT conversion requires CUDA.")

    log(f"PyTracking root: {pytracking_root}")
    log(f"Checkpoint: {checkpoint}")
    log(f"Output dir: {output_dir}")
    log(f"Input size: {args.height}x{args.width}")
    log(f"Batches: {args.batches}")
    log(f"Precision: {args.precision}")

    log("Loading DiMP50 network for weights/architecture...")

    net, _ = ltr_loading.load_network(
        network_dir=str(checkpoint),
        backbone_pretrained=False,
    )

    net.eval().to(device)

    backbone = net.feature_extractor.eval().to(device)
    trunk = ResNetTrunk(backbone).eval().to(device)

    # Sanity forward
    x = torch.randn(1, 3, args.height, args.width, device=device)

    with torch.no_grad():
        y = trunk(x)

    log("PyTorch trunk output shapes:")
    for name, tensor in zip(OUT_LAYERS, y):
        log(f"  {name}: {tuple(tensor.shape)}")

    use_fp16 = args.precision == "fp16"

    for batch_size in args.batches:
        out_path = output_dir / (
            f"dimp50_trunk_B{batch_size}_3x{args.height}x{args.width}_{args.precision}_trt.pth"
        )

        log("")
        log(
            f"Converting DiMP50 trunk: "
            f"input=[{batch_size},3,{args.height},{args.width}], "
            f"precision={args.precision}"
        )

        inp = torch.randn(batch_size, 3, args.height, args.width, device=device)

        t0 = time.time()

        with torch.no_grad():
            trt_mod = torch2trt(
                trunk,
                [inp],
                fp16_mode=use_fp16,
                int8_mode=False,
                max_workspace_size=args.workspace,
                max_batch_size=batch_size,
                log_level=trt.Logger.WARNING,
                default_device_type=trt.DeviceType.GPU,
            )

        torch.cuda.synchronize()
        log(f"Convert time: {time.time() - t0:.2f}s")

        # Accuracy check against PyTorch
        test_inp = torch.randn(batch_size, 3, args.height, args.width, device=device)

        with torch.no_grad():
            y_pt = trunk(test_inp)
            y_trt = trt_mod(test_inp)

        torch.cuda.synchronize()

        log("PyTorch vs TensorRT output differences:")
        for name, a, b in zip(OUT_LAYERS, y_pt, y_trt):
            a = a.float()
            b = b.float()

            abs_diff = (a - b).abs()
            max_abs = abs_diff.max().item()
            mean_abs = abs_diff.mean().item()
            max_rel = (abs_diff / a.abs().clamp_min(1e-6)).max().item()

            log(
                f"  {name}: "
                f"max_abs={max_abs:.6e}, "
                f"mean_abs={mean_abs:.6e}, "
                f"max_rel={max_rel:.6e}"
            )

        # Warmup
        with torch.no_grad():
            for _ in range(20):
                _ = trt_mod(inp)

        torch.cuda.synchronize()

        save_bundle(
            out_path,
            trt_mod.state_dict(),
            OUT_LAYERS,
            (batch_size, 3, args.height, args.width),
            args.precision,
        )

    log("Done.")


if __name__ == "__main__":
    main()
