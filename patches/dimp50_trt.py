import os

from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone


def _env_bool(name, default=False):
    value = os.environ.get(name, str(int(default))).lower()
    return value in ("1", "true", "yes", "y", "on")


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    # Must match TensorRT backbone input size.
    # If this is changed to 256, rebuild TRT engines for 3x256x256.
    params.image_sample_size = 18 * 16  # 288
    params.search_area_scale = 4

    # -------------------------------------------------------------------------
    # Learning parameters
    # -------------------------------------------------------------------------
    params.sample_memory_size = 25
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25

    # Higher value = less frequent classifier updates = faster FPS.
    params.train_skipping = 20

    # -------------------------------------------------------------------------
    # Classifier optimization parameters
    # -------------------------------------------------------------------------
    params.update_classifier = True

    # Jetson fast-init defaults.
    # Original DiMP50 commonly uses higher values, but they make target init slow.
    params.net_opt_iter = int(os.environ.get("DIMP50_NET_OPT_ITER", "3"))
    params.net_opt_update_iter = int(os.environ.get("DIMP50_NET_OPT_UPDATE_ITER", "1"))
    params.net_opt_hn_iter = int(os.environ.get("DIMP50_NET_OPT_HN_ITER", "0"))

    # -------------------------------------------------------------------------
    # Detection parameters
    # -------------------------------------------------------------------------
    params.window_output = False

    # -------------------------------------------------------------------------
    # Init augmentation parameters
    # -------------------------------------------------------------------------
    # Default: disabled for fastest object-selection initialization.
    # Enable with:
    #   DIMP50_USE_AUG=1 python run_video.py dimp dimp50 ...
    use_aug = _env_bool("DIMP50_USE_AUG", default=False)

    params.use_augmentation = use_aug

    if use_aug:
        # Lightweight augmentation: much faster than original rotate/blur/dropout set.
        params.augmentation = {
            "fliplr": True,
            "relativeshift": [(0.15, 0.15), (-0.15, 0.15)],
        }
    else:
        params.augmentation = {}

    params.augmentation_expansion_factor = 1
    params.random_shift_factor = 1 / 3

    # -------------------------------------------------------------------------
    # Advanced localization parameters
    # -------------------------------------------------------------------------
    params.advanced_localization = True
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2

    # Keep the original PyTracking spelling if tracker code expects it.
    params.dispalcement_scale = 0.8

    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    # -------------------------------------------------------------------------
    # IoUNet parameters
    # -------------------------------------------------------------------------
    params.iounet_augmentation = False
    params.iounet_use_log_scale = True
    params.iounet_k = 3

    # Fast-init defaults.
    params.num_init_random_boxes = int(os.environ.get("DIMP50_NUM_INIT_RANDOM_BOXES", "5"))
    params.box_jitter_pos = 0.1
    params.box_jitter_sz = 0.5
    params.maximal_aspect_ratio = 6
    params.box_refinement_iter = int(os.environ.get("DIMP50_BOX_REFINEMENT_ITER", "2"))
    params.box_refinement_step_length = 1
    params.box_refinement_step_decay = 1

    # Runtime checkpoint where feature_extractor.* weights are removed.
    # The patched loader injects TensorRT backbone modules at runtime.
    runtime_ckpt = os.environ.get(
        "DIMP50_RUNTIME_CKPT",
        "/home/jetson/Desktop/pytracking/conversion/dimp50_runtime.pth",
    )

    params.net = NetWithBackbone(
        net_path=runtime_ckpt,
        use_gpu=params.use_gpu,
    )

    params.vot_anno_conversion_type = "preserve_area"

    return params
