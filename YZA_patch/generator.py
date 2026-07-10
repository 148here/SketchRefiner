#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Online image/edge/sketch/mask generation for SketchRefiner."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import importlib
import importlib.util
import os
import sys
import types

import numpy as np
from PIL import Image

# Edge-JointDiT/YZApatch still uses deprecated numpy aliases. The training
# runtime that can import SketchInpainter Stage3 uses a newer numpy, so provide
# the old aliases before importing those modules.
for _alias_name, _alias_value in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
):
    if _alias_name not in np.__dict__:
        setattr(np, _alias_name, _alias_value)

from .config import (
    MASK_PARAMS,
    RESOLUTION,
    SKETCH_BACKEND,
    SKETCH_PARAMS,
    USE_COMPLEX_MASK,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _codes_root() -> Path:
    return _project_root().parent


def _import_yzapatch_modules() -> Dict[str, Any]:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[1]
    yzapatch_dir = project_root / "diffusers" / "examples" / "controlnet" / "YZApatch"

    if not yzapatch_dir.exists():
        raise ImportError(
            "Cannot find YZApatch at %s. Expected it under "
            "codes/diffusers/examples/controlnet/YZApatch." % yzapatch_dir
        )

    if str(yzapatch_dir) not in sys.path:
        sys.path.insert(0, str(yzapatch_dir))

    try:
        from config import (  # type: ignore
            DEXINED_CHECKPOINT,
            DEXINED_DEVICE,
            DEXINED_THRESHOLD,
            EDGE_CACHE_DIR,
            EDGE_CACHE_VERSION,
            IMAGE_EXTENSIONS,
            MASK_PARAMS as YZAPATCH_MASK_PARAMS,
            SKETCH_PARAMS as YZAPATCH_SKETCH_PARAMS,
            SKETCH_UTIL_DIR,
        )
        from edge_cache import get_edge_cache_manager  # type: ignore
    except ImportError as e:
        raise ImportError("Failed to import YZApatch config/modules: %s" % e)

    if SKETCH_UTIL_DIR not in sys.path:
        sys.path.insert(0, SKETCH_UTIL_DIR)
    try:
        from dataset.sketch_util import (  # type: ignore
            extract_edge,
            make_sketch_from_image_or_edge,
        )
    except ImportError as e:
        raise ImportError(
            "Failed to import dataset.sketch_util from %s: %s"
            % (SKETCH_UTIL_DIR, e)
        )

    return {
        "SKETCH_UTIL_DIR": SKETCH_UTIL_DIR,
        "DEXINED_CHECKPOINT": DEXINED_CHECKPOINT,
        "DEXINED_THRESHOLD": DEXINED_THRESHOLD,
        "DEXINED_DEVICE": DEXINED_DEVICE,
        "EDGE_CACHE_DIR": EDGE_CACHE_DIR,
        "YZAPATCH_SKETCH_PARAMS": YZAPATCH_SKETCH_PARAMS,
        "YZAPATCH_MASK_PARAMS": YZAPATCH_MASK_PARAMS,
        "IMAGE_EXTENSIONS": IMAGE_EXTENSIONS,
        "EDGE_CACHE_VERSION": EDGE_CACHE_VERSION,
        "get_edge_cache_manager": get_edge_cache_manager,
        "make_sketch_from_image_or_edge": make_sketch_from_image_or_edge,
        "extract_edge": extract_edge,
    }


_MODULES_CACHE: Optional[Dict[str, Any]] = None
_SKETCHINPAINTER_CACHE: Optional[Dict[str, Any]] = None


def _get_modules() -> Dict[str, Any]:
    global _MODULES_CACHE
    if _MODULES_CACHE is None:
        _MODULES_CACHE = _import_yzapatch_modules()
    return _MODULES_CACHE


def _ensure_alias_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = module
    else:
        module.__path__ = [str(path)]  # type: ignore[attr-defined]


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load module %s from %s" % (module_name, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_sketchinpainter_root() -> Path:
    env_root = os.environ.get("SKETCHINPAINTER_ROOT", "").strip()
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(_codes_root() / "SketchInpainter")

    for candidate in candidates:
        if (candidate / "dataset" / "makesketch").is_dir():
            return candidate

    raise ImportError(
        "Cannot find SketchInpainter. Set SKETCHINPAINTER_ROOT or place it "
        "beside SketchRefiner under the same codes directory."
    )


def _import_sketchinpainter_stage3() -> Dict[str, Any]:
    global _SKETCHINPAINTER_CACHE
    if _SKETCHINPAINTER_CACHE is not None:
        return _SKETCHINPAINTER_CACHE

    root = _find_sketchinpainter_root()

    # Import dataset.makesketch through an alias package so it cannot collide
    # with YZApatch's top-level "dataset" package.
    dataset_alias = "_sketchinpainter_dataset"
    _ensure_alias_package(dataset_alias, root / "dataset")
    makesketch = importlib.import_module(dataset_alias + ".makesketch")

    data_defaults_path = root / "configs" / "stage3" / "data_defaults.py"
    if not data_defaults_path.is_file():
        raise ImportError("Missing SketchInpainter Stage3 defaults: %s" % data_defaults_path)
    data_defaults = _load_module_from_path(
        "_sketchinpainter_stage3_data_defaults",
        data_defaults_path,
    )

    config_path = getattr(data_defaults, "SKETCH_CONFIG_PATH", "")
    if not config_path or not os.path.isfile(config_path):
        config_path = str(root / "dataset" / "makesketch" / "config.py")

    _SKETCHINPAINTER_CACHE = {
        "root": root,
        "make_sketch_from_edge": getattr(makesketch, "make_sketch_from_edge"),
        "config_path": config_path,
        "sketch_overrides": dict(getattr(data_defaults, "SKETCH_CONFIG_OVERRIDES", {}) or {}),
    }
    return _SKETCHINPAINTER_CACHE


def _effective_resolution(config_resolution: Optional[int], fallback: int) -> int:
    if config_resolution is not None and int(config_resolution) > 0:
        return int(config_resolution)
    return int(fallback)


def _load_and_resize_image(image_path: str, resolution: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    if img.size != (resolution, resolution):
        resample = Image.BILINEAR if resolution == 256 else Image.LANCZOS
        img = img.resize((resolution, resolution), resample)
    return np.array(img, dtype=np.uint8)


def _build_extract_fn(modules: Dict[str, Any]):
    extract_edge = modules["extract_edge"]

    def _extract_fn(img_np: np.ndarray) -> np.ndarray:
        return extract_edge(
            image=img_np,
            method="dexined",
            dexined_checkpoint=modules["DEXINED_CHECKPOINT"],
            dexined_threshold=modules["DEXINED_THRESHOLD"],
            dexined_device=modules["DEXINED_DEVICE"],
        )

    return _extract_fn


def _get_edge_image(
    image_np: np.ndarray,
    image_path: str,
    modules: Dict[str, Any],
    enable_cache: bool = True,
) -> np.ndarray:
    extract_fn = _build_extract_fn(modules)
    if enable_cache:
        dexined_params = {
            "threshold": modules["DEXINED_THRESHOLD"],
            "version": modules["EDGE_CACHE_VERSION"],
        }
        cache_manager = modules["get_edge_cache_manager"](
            modules["EDGE_CACHE_DIR"],
            dexined_params,
        )
        return cache_manager.get_or_compute_edge(
            image_path=str(image_path),
            image_np=image_np,
            enable_cache=True,
            extract_fn=extract_fn,
        )
    return extract_fn(image_np)


def _merge_params(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def _make_sketch_with_yzapatch(
    edge_np: np.ndarray,
    modules: Dict[str, Any],
    sketch_params: Optional[Dict[str, Any]],
) -> np.ndarray:
    sp = _merge_params(modules["YZAPATCH_SKETCH_PARAMS"], sketch_params or SKETCH_PARAMS)
    seed = np.random.randint(0, 2**31 - 1)
    return modules["make_sketch_from_image_or_edge"](
        input_image=edge_np,
        seed=int(seed),
        is_edge=True,
        enable_edge_extraction=False,
        sigma_mean=sp.get("sigma_mean", 13.0),
        sigma_std=sp.get("sigma_std", 2.6),
        spatial_smooth_sigma=sp.get("spatial_smooth_sigma", 2.0),
        cp_sigma_mean=sp.get("cp_sigma_mean", 2.1),
        cp_sigma_std=sp.get("cp_sigma_std", 0.4),
        cp_spatial_smooth=sp.get("cp_spatial_smooth", 1.5),
    )


def _make_sketch_with_sketchinpainter_stage3(
    edge_np: np.ndarray,
    sketch_params: Optional[Dict[str, Any]],
    mask_np: Optional[np.ndarray] = None,
) -> np.ndarray:
    stage3 = _import_sketchinpainter_stage3()
    overrides = _merge_params(stage3["sketch_overrides"], sketch_params or SKETCH_PARAMS)
    seed = np.random.randint(0, 2**31 - 1)
    mask_mode = "mask_region" if mask_np is not None else "full_image"
    return stage3["make_sketch_from_edge"](
        edge_np,
        seed=int(seed),
        config_path=stage3["config_path"],
        mask=mask_np,
        mask_mode=mask_mode,
        boundary_pin_px=12.0 if mask_np is not None else 0.0,
        **overrides
    )


def clear_edge_cache() -> None:
    try:
        modules = _get_modules()
        dexined_params = {
            "threshold": modules["DEXINED_THRESHOLD"],
            "version": modules["EDGE_CACHE_VERSION"],
        }
        manager = modules["get_edge_cache_manager"](modules["EDGE_CACHE_DIR"], dexined_params)
        manager.clear_cache()
    except Exception:
        pass


def generate_triplet(
    image_path: str,
    resolution: Optional[int] = None,
    use_complex_mask: Optional[bool] = None,
    sketch_params: Optional[Dict[str, Any]] = None,
    mask_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    modules = _get_modules()
    effective_res = _effective_resolution(RESOLUTION if resolution is None else resolution, fallback=256)
    mask_flag = USE_COMPLEX_MASK if use_complex_mask is None else use_complex_mask

    image_np = _load_and_resize_image(image_path, effective_res)
    edge_np = _get_edge_image(image_np, image_path, modules, enable_cache=True)

    mask_np: Optional[np.ndarray] = None
    if mask_flag:
        # Keep the original complex-mask path for the legacy YZApatch workflow.
        import importlib as _importlib

        yza_config = _importlib.import_module("config")
        mask_generator_module = _importlib.import_module("mask_generator")
        base_mask_params = getattr(yza_config, "MASK_PARAMS")
        mp = _merge_params(base_mask_params, mask_params or MASK_PARAMS)
        mask_generator = mask_generator_module.ComplexMaskGenerator(mp)
        mask_seed = np.random.randint(0, 2**31 - 1)
        mask_np = mask_generator.generate(edge_np, seed=int(mask_seed))

    backend = str(SKETCH_BACKEND or "yzapatch").strip().lower()
    if backend == "sketchinpainter_stage3":
        sketch_np = _make_sketch_with_sketchinpainter_stage3(
            edge_np=edge_np,
            sketch_params=sketch_params,
            mask_np=mask_np,
        )
    elif backend == "yzapatch":
        sketch_np = _make_sketch_with_yzapatch(
            edge_np=edge_np,
            modules=modules,
            sketch_params=sketch_params,
        )
    else:
        raise ValueError("Unsupported SKETCH_BACKEND: %s" % SKETCH_BACKEND)

    return image_np, edge_np, sketch_np, mask_np
