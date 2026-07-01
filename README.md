# PyTracking DiMP50 TensorRT on NVIDIA Jetson

This repository documents and provides scripts/patches for running the PyTracking DiMP tracker on NVIDIA Jetson with a TensorRT-accelerated DiMP50 backbone.

The goal is to improve model loading and tracking speed by replacing the PyTorch ResNet50 backbone inside DiMP50 with torch2trt TensorRT modules.

## Project Status

Implemented:

* PyTracking + LTR installation on Jetson using Python venv
* torch2trt installation
* DiMP50 checkpoint loading
* DiMP50 backbone trunk conversion to TensorRT
* B=1 and B=13 TensorRT backbone engines
* Runtime checkpoint creation with backbone weights removed
* Patched LTR loader for non-strict loading and TRT backbone injection
* Patched PyTracking loader to use TensorRT backbone bundles
* Video/webcam tracking with DiMP50 + TRT backbone

Current observed performance:

* DiMP50 + TensorRT backbone runs at approximately 7 FPS on Jetson in the tested setup.
* Model loading is faster when using a slim runtime checkpoint and prebuilt TensorRT backbone bundles.

## Repository Layout

Expected working layout on Jetson:

```text
~/Desktop/pytracking/
├── env/                         # Python virtual environment, not committed
├── pytracking/                  # visionml/pytracking repository
├── torch2trt/                   # NVIDIA-AI-IOT/torch2trt repository
├── conversion/
│   ├── conversion_dimp50.py
│   ├── create_runtime_ckpt.py
│   ├── dimp50.pth               # not committed
│   ├── dimp50_runtime.pth       # not committed
│   ├── dimp50_trunk_B1_3x288x288_fp16_trt.pth    # not committed
│   └── dimp50_trunk_B13_3x288x288_fp16_trt.pth   # not committed
└── install_pytracking_jetson.sh
```

This GitHub repository stores the scripts and patches, not the environment, video files, model checkpoints, or TensorRT engine files.

## Hardware / Software Assumptions

Tested target:

* NVIDIA Jetson
* JetPack with CUDA and TensorRT installed system-wide
* Python 3.10
* PyTorch and TorchVision installed from Jetson-compatible wheels
* torch2trt built locally
* PyTracking / LTR tracker code

The install script uses:

* Python venv
* `--system-site-packages` so system TensorRT is visible inside the venv
* local PyTorch and TorchVision wheels
* torch2trt editable install
* optional PreciseRoIPooling build

## Installation

Clone or place the project in this layout:

```bash
cd ~/Desktop
mkdir pytracking
cd pytracking
```

Expected subfolders:

```text
pytracking/      # actual PyTracking repository
torch2trt/       # torch2trt repository
conversion/      # conversion scripts
```

Run:

```bash
bash install_pytracking_jetson.sh /home/jetson/Desktop/pytracking
```

The installer will:

1. Install required apt packages.
2. Create a Python venv.
3. Install PyTorch and TorchVision wheels.
4. Install PyTracking Python dependencies.
5. Initialize PyTracking submodules.
6. Build PreciseRoIPooling if available.
7. Install torch2trt.
8. Create local PyTracking/LTR environment files.
9. Optionally download the DiMP50 checkpoint.

Activate the environment:

```bash
cd ~/Desktop/pytracking
source env/bin/activate
```

## Quick Start from This Repository

This repository is not a fork of PyTracking. It is a Jetson integration layer containing install scripts, TensorRT conversion scripts, and patch files.

A fresh setup uses two folders:

```text
~/Desktop/pytracking-jetson-trt/   # this repository
~/Desktop/pytracking/              # generated runnable workspace
```

## Important Compatibility Patch

Older PyTracking/LTR code may fail with newer TorchVision because this import was removed:

```python
from torchvision.models.resnet import model_urls
```

Patch:

```python
try:
    from torchvision.models.resnet import model_urls
except ImportError:
    model_urls = {}
```

This is usually located in:

```text
pytracking/ltr/models/backbone/resnet.py
```

## DiMP50 Backbone TensorRT Conversion

The DiMP50 backbone is ResNet50. We convert the ResNet trunk up to `layer4` and return:

```text
layer1
layer2
layer3
layer4
```

The conversion script builds two TensorRT bundles:

```text
B=1,  3x288x288
B=13, 3x288x288
```

Why two engines?

* `B=1` is used during normal tracking.
* `B=13` is used during DiMP initialization when augmentation samples are processed together.

Run:

```bash
cd ~/Desktop/pytracking
source env/bin/activate

python conversion/conversion_dimp50.py
```

Expected outputs:

```text
conversion/dimp50_trunk_B1_3x288x288_fp16_trt.pth
conversion/dimp50_trunk_B13_3x288x288_fp16_trt.pth
```

If you change:

```python
params.image_sample_size
```

from `288` to another value such as `256`, you must rebuild the TensorRT engines for that input size.

## Runtime Checkpoint

The TensorRT backbone replaces the PyTorch `feature_extractor`, so the runtime checkpoint does not need the original ResNet50 backbone weights.

Create a smaller runtime checkpoint:

```bash
python conversion/create_runtime_ckpt.py \
  --in conversion/dimp50.pth \
  --out conversion/dimp50_runtime.pth \
  --drop_prefix feature_extractor. \
  --strip_optimizer
```

This removes all state dict keys starting with:

```text
feature_extractor.
```

Important: this checkpoint cannot be loaded with strict PyTorch loading unless the loader is patched to inject the TRT backbone.

## Loader Patch

The standard LTR loader uses strict loading:

```python
net.load_state_dict(checkpoint_dict["net"])
```

That fails for runtime checkpoints because `feature_extractor.*` keys have been removed.

The patched loader must:

1. Load the non-backbone weights with `strict=False`.
2. Skip missing `feature_extractor.*` keys.
3. Load TensorRT backbone bundles.
4. Replace `net.feature_extractor` with a batch router.

Conceptually:

```python
missing, unexpected = net.load_state_dict(state, strict=False)
net.feature_extractor = TRTBatchRouter(backbone_b1, backbone_b13)
```

The `TRTBatchRouter` chooses the correct TensorRT engine based on input batch size:

```python
if B == 1:
    use B1 engine
elif B == 13:
    use B13 engine
else:
    raise error or chunk into B=1 calls
```

## PyTracking Loader Patch

The PyTracking utility loader must call the patched LTR loader with TensorRT paths:

```python
net, _ = ltr_loading.load_network(
    path_full,
    trt_backbone_path={
        1:  "/home/jetson/Desktop/pytracking/conversion/dimp50_trunk_B1_3x288x288_fp16_trt.pth",
        13: "/home/jetson/Desktop/pytracking/conversion/dimp50_trunk_B13_3x288x288_fp16_trt.pth",
    },
    trt_backbone_attr="feature_extractor",
    trt_device="cuda",
    trt_layer_names=("layer1", "layer2", "layer3", "layer4"),
    trt_cls_layer="layer3",
    skip_backbone_weights=True,
    verbose=True,
    **kwargs
)
```

If this is not done, loading `dimp50_runtime.pth` will fail with missing `feature_extractor.*` keys.

## DiMP50 Parameter File

In the DiMP50 parameter file, set the network path to the runtime checkpoint:

```python
params.net = NetWithBackbone(
    net_path="/home/jetson/Desktop/pytracking/conversion/dimp50_runtime.pth",
    use_gpu=params.use_gpu
)
```

Recommended speed-oriented initialization settings:

```python
params.net_opt_iter = 3
params.net_opt_update_iter = 1
params.net_opt_hn_iter = 0

params.box_refinement_iter = 2
params.num_init_random_boxes = 5
```

For faster object selection initialization, reduce augmentation:

```python
params.use_augmentation = True
params.augmentation = {
    "fliplr": True,
    "relativeshift": [(0.15, 0.15), (-0.15, 0.15)]
}
params.augmentation_expansion_factor = 1
```

Maximum speed option:

```python
params.use_augmentation = False
params.augmentation = {}
```

This is faster but less robust.

## Sanity Test

Before running video/webcam tracking, test the TRT backbone:

```bash
python conversion/test_trt_backbone.py
```

Expected DiMP50 `layer3` shape:

```text
torch.Size([1, 1024, 18, 18])
```

If you see a TensorRT shape mismatch such as:

```text
Expected [1,3,288,288] but got [1,3,256,256]
```

then `params.image_sample_size` does not match the TensorRT engine input size. Rebuild the engines or restore `image_sample_size = 288`.

## Run Tracking

Video:

```bash
cd ~/Desktop/pytracking/pytracking/pytracking
python run_video.py dimp dimp50 ../../1.mp4
```

Webcam:

```bash
cd ~/Desktop/pytracking/pytracking/pytracking
python run_webcam.py dimp dimp50
```

## Troubleshooting

### Missing `feature_extractor.*` keys

Cause:

* You are using `dimp50_runtime.pth`
* But the loader is still doing strict `load_state_dict`

Fix:

* Use the patched LTR loader.
* Pass `trt_backbone_path`.
* Use `strict=False`.
* Inject TensorRT backbone router.

### TensorRT static shape mismatch

Example:

```text
Expected [1,3,288,288], got [1,3,256,256]
```

Cause:

* TensorRT engine was built for 288x288.
* Runtime is feeding 256x256.

Fix:

* Set `params.image_sample_size = 288`, or
* rebuild B1/B13 TensorRT engines for 256x256.

### TensorRT engine warning: different device

Example:

```text
Using an engine plan file across different models of devices is not recommended
```

Fix:

* Rebuild TensorRT bundles on the exact Jetson device where they will run.

### Slow initialization after selecting object

Main causes:

* many augmentation samples
* high `net_opt_iter`
* box refinement iterations

Recommended fast init:

```python
params.net_opt_iter = 3
params.net_opt_update_iter = 1
params.net_opt_hn_iter = 0
params.box_refinement_iter = 2
params.num_init_random_boxes = 5
```

### Slow cold start

Helpful steps:

* use `dimp50_runtime.pth`
* avoid committing/loading large full checkpoints unnecessarily
* prebuild PreciseRoIPooling
* keep TensorRT engines on fast storage
* run `sudo nvpmodel -m 0` and `sudo jetson_clocks` for performance mode when appropriate

## Files Not Included

The following files are intentionally not committed:

```text
env/
*.pth
*.pth.tar
*.onnx
*.engine
*.mp4
*.avi
output_video.*
```

Model files and TensorRT bundles should be generated locally or distributed separately through release assets or Git LFS.

## License Notes

This project is based on PyTracking and torch2trt. Check and respect the licenses of the upstream projects before redistribution.

## TODO

* Add classifier.feature_extractor TensorRT conversion
* Add automatic benchmark script
* Add dynamic/chunking fallback for unsupported batch sizes
* Add setup script to apply patches automatically
* Add FPS comparison table for PyTorch vs TensorRT backbone

