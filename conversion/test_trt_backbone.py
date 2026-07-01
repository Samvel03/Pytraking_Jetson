""""
python conversion/test_trtr_backbone.py \
  --pytracking-root /home/jetson/Desktop/pytracking/pytracking \
  --checkpoint /home/jetson/Desktop/pytracking/conversion/dimp50_runtime.pth \
  --trt-b1 /home/jetson/Desktop/pytracking/conversion/dimp50_trunk_B1_3x288x288_fp16_trt.pth \
  --trt-b13 /home/jetson/Desktop/pytracking/conversion/dimp50_trunk_B13_3x288x288_fp16_trt.pth
"""
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test DiMP50 runtime checkpoint + TensorRT backbone injection."
    )

    parser.add_argument(
        "--pytracking-root",
        default="/home/jetson/Desktop/pytracking/pytracking",
        help="Path to actual PyTracking repository containing ltr/ and pytracking/.",
    )

    parser.add_argument(
        "--checkpoint",
        default="/home/jetson/Desktop/pytracking/conversion/dimp50_runtime.pth",
        help="Path to dimp50_runtime.pth.",
    )

    parser.add_argument(
        "--trt-b1",
        default="/home/jetson/Desktop/pytracking/conversion/dimp50_trunk_B1_3x288x288_fp16_trt.pth",
        help="Path to B=1 TensorRT backbone bundle.",
    )

    parser.add_argument(
        "--trt-b13",
        default="/home/jetson/Desktop/pytracking/conversion/dimp50_trunk_B13_3x288x288_fp16_trt.pth",
        help="Path to B=13 TensorRT backbone bundle.",
    )

    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=288)

    return parser.parse_args()


def main():
    args = parse_args()

    pytracking_root = Path(args.pytracking_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    trt_b1 = Path(args.trt_b1).resolve()
    trt_b13 = Path(args.trt_b13).resolve()

    if not pytracking_root.exists():
        raise FileNotFoundError(f"PyTracking root does not exist: {pytracking_root}")

    if not (pytracking_root / "ltr").exists():
        raise FileNotFoundError(
            f"Invalid PyTracking root: {pytracking_root}. Expected ltr/ inside it."
        )

    if not checkpoint.exists():
        raise FileNotFoundError(f"Runtime checkpoint does not exist: {checkpoint}")

    if not trt_b1.exists():
        raise FileNotFoundError(f"B1 TRT bundle does not exist: {trt_b1}")

    if not trt_b13.exists():
        raise FileNotFoundError(f"B13 TRT bundle does not exist: {trt_b13}")

    sys.path.insert(0, str(pytracking_root))

    import ltr.admin.loading as ltr_loading

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    net, _ = ltr_loading.load_network(
        network_dir=str(checkpoint),
        trt_backbone_path={
            1: str(trt_b1),
            13: str(trt_b13),
        },
        trt_backbone_attr="feature_extractor",
        trt_device="cuda",
        trt_layer_names=("layer1", "layer2", "layer3", "layer4"),
        trt_cls_layer="layer3",
        skip_backbone_weights=True,
        verbose=True,
    )

    net.eval().cuda()

    with torch.no_grad():
        x1 = torch.randn(1, 3, args.height, args.width, device="cuda")
        out1 = net.feature_extractor(x1, layers=["layer1", "layer2", "layer3", "layer4"])

        print("B=1 keys:", out1.keys())
        print("B=1 layer1:", tuple(out1["layer1"].shape))
        print("B=1 layer2:", tuple(out1["layer2"].shape))
        print("B=1 layer3:", tuple(out1["layer3"].shape))
        print("B=1 layer4:", tuple(out1["layer4"].shape))

        x13 = torch.randn(13, 3, args.height, args.width, device="cuda")
        out13 = net.feature_extractor(x13, layers=["layer1", "layer2", "layer3", "layer4"])

        print("B=13 keys:", out13.keys())
        print("B=13 layer1:", tuple(out13["layer1"].shape))
        print("B=13 layer2:", tuple(out13["layer2"].shape))
        print("B=13 layer3:", tuple(out13["layer3"].shape))
        print("B=13 layer4:", tuple(out13["layer4"].shape))

    print("OK ✅")


if __name__ == "__main__":
    main()
