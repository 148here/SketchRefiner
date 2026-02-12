#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用 edge / sketch / mask 在线生成工具
======================================

设计目标：
- 为 SketchRefiner 的 SRN 提供一套与 diffusers/examples/controlnet/YZApatch
  一致的数据预处理逻辑；
- 尽量复用 YZApatch 中现有的 DexiNed + sketch_util + ComplexMaskGenerator；
- 将本仓库特有的分辨率 / 覆盖参数放在 YZA_patch/config.py 中集中管理。

核心入口：
- generate_triplet(image_path, resolution=None, use_complex_mask=None, sketch_params=None, mask_params=None)
"""

# from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import sys

from .config import RESOLUTION, USE_COMPLEX_MASK, SKETCH_PARAMS, MASK_PARAMS


def _import_yzapatch_modules() -> Dict[str, Any]:
    """
    动态导入 diffusers/examples/controlnet/YZApatch 中的配置和工具。

    返回一个字典，包含：
    - SKETCH_UTIL_DIR, DEXINED_CHECKPOINT, DEXINED_THRESHOLD, DEXINED_DEVICE,
      EDGE_CACHE_DIR, SKETCH_PARAMS, MASK_PARAMS, IMAGE_EXTENSIONS, EDGE_CACHE_VERSION
    - get_edge_cache_manager, make_sketch_from_image_or_edge, extract_edge
    """
    current_dir = Path(__file__).resolve().parent
    # 当前目录结构：yza/SketchRefiner/YZA_patch
    # 项目根目录：yza（向上1级）
    project_root = current_dir.parents[1]
    yzapatch_dir = project_root / "diffusers" / "examples" / "controlnet" / "YZApatch"

    if not yzapatch_dir.exists():
        raise ImportError(
            f"无法找到 YZApatch 目录：{yzapatch_dir}，"
            "请确认 diffusers 子模块已正确放置在 codes/diffusers/examples/controlnet/YZApatch 下。"
        )

    if str(yzapatch_dir) not in sys.path:
        sys.path.insert(0, str(yzapatch_dir))

    try:
        from config import (  # type: ignore
            SKETCH_UTIL_DIR,
            DEXINED_CHECKPOINT,
            DEXINED_THRESHOLD,
            DEXINED_DEVICE,
            EDGE_CACHE_DIR,
            SKETCH_PARAMS as YZAPATCH_SKETCH_PARAMS,
            MASK_PARAMS as YZAPATCH_MASK_PARAMS,
            IMAGE_EXTENSIONS,
            EDGE_CACHE_VERSION,
        )
        from edge_cache import get_edge_cache_manager  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"无法从 YZApatch 导入配置/模块，请检查目录结构和 PYTHONPATH：{e}"
        )

    # 导入 sketch_util
    if SKETCH_UTIL_DIR not in sys.path:
        sys.path.insert(0, SKETCH_UTIL_DIR)
    try:
        from dataset.sketch_util import (  # type: ignore
            make_sketch_from_image_or_edge,
            extract_edge,
        )
    except ImportError as e:
        raise ImportError(
            f"无法从 SKETCH_UTIL_DIR 导入 dataset.sketch_util，"
            f"请检查 YZApatch.config.SKETCH_UTIL_DIR={SKETCH_UTIL_DIR}：{e}"
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


def _get_modules() -> Dict[str, Any]:
    global _MODULES_CACHE
    if _MODULES_CACHE is None:
        _MODULES_CACHE = _import_yzapatch_modules()
    return _MODULES_CACHE


def _effective_resolution(config_resolution: Optional[int], fallback: int) -> int:
    """
    计算实际使用的分辨率：
    - 优先使用 config.py 中的 RESOLUTION；
    - 若无效，则回退到调用方给定的 fallback（通常是 configs.size）。
    """
    if config_resolution is not None and int(config_resolution) > 0:
        return int(config_resolution)
    return int(fallback)


def _load_and_resize_image(image_path: str, resolution: int) -> np.ndarray:
    """
    加载并 resize 原图到指定分辨率，返回 [H,W,3] 的 uint8 RGB 数组。

    插值策略：
    - 分辨率 256：使用「线性插值」风格（BILINEAR），更贴近 SRN 侧使用习惯；
    - 分辨率 512：使用与 SDXL / YZApatch 一致的 LANCZOS；
    - 其他分辨率：也使用 LANCZOS。
    """
    img = Image.open(image_path).convert("RGB")

    if img.size != (resolution, resolution):
        if resolution == 256:
            img = img.resize((resolution, resolution), Image.BILINEAR)
        elif resolution == 512:
            img = img.resize((resolution, resolution), Image.LANCZOS)
        else:
            img = img.resize((resolution, resolution), Image.LANCZOS)

    return np.array(img, dtype=np.uint8)


def _build_extract_fn(modules: Dict[str, Any]):
    """
    构造与 InpaintingSketchDataset._extract_edge_with_cache 一致的边缘提取函数。
    """
    extract_edge = modules["extract_edge"]
    dexined_checkpoint = modules["DEXINED_CHECKPOINT"]
    dexined_threshold = modules["DEXINED_THRESHOLD"]
    dexined_device = modules["DEXINED_DEVICE"]

    def _extract_fn(img_np: np.ndarray) -> np.ndarray:
        return extract_edge(
            image=img_np,
            method="dexined",
            dexined_checkpoint=dexined_checkpoint,
            dexined_threshold=dexined_threshold,
            dexined_device=dexined_device,
        )

    return _extract_fn


def _get_edge_image(
    image_np: np.ndarray,
    image_path: str,
    modules: Dict[str, Any],
    enable_cache: bool = True,
) -> np.ndarray:
    """
    获取 edge 图：
    - 若启用缓存，则使用 YZApatch.edge_cache.EdgeCacheManager；
    - 否则直接调用 DexiNed 提取。
    """
    extract_fn = _build_extract_fn(modules)
    if enable_cache:
        get_edge_cache_manager = modules["get_edge_cache_manager"]
        dexined_params = {
            "threshold": modules["DEXINED_THRESHOLD"],
            "version": modules["EDGE_CACHE_VERSION"],
        }
        cache_manager = get_edge_cache_manager(modules["EDGE_CACHE_DIR"], dexined_params)
        edge_image = cache_manager.get_or_compute_edge(
            image_path=str(image_path),
            image_np=image_np,
            enable_cache=True,
            extract_fn=extract_fn,
        )
    else:
        edge_image = extract_fn(image_np)

    return edge_image


def _merge_params(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    将本地 config 中的覆盖参数与 YZApatch.config 默认参数合并：
    - override 为空：返回 base；
    - override 非空：base.update(override) 后返回。
    """
    if not override:
        return dict(base)
    merged = dict(base)
    merged.update(override)
    return merged


def generate_triplet(
    image_path: str,
    resolution: Optional[int] = None,
    use_complex_mask: Optional[bool] = None,
    sketch_params: Optional[Dict[str, Any]] = None,
    mask_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    对单张图片执行完整的数据预处理流程：
    1. 加载原图并 resize 到目标分辨率；
    2. 提取/缓存 edge 图；
    3. 基于 edge 生成 sketch；
    4. （可选）基于 edge 生成复杂 mask。

    Args:
        image_path: 原图路径
        resolution: 目标分辨率；若为 None，则使用 config.RESOLUTION
        use_complex_mask: 是否生成复杂 mask；若为 None，则使用 config.USE_COMPLEX_MASK
        sketch_params: 覆盖 sketch 生成参数；None 时使用 config.SKETCH_PARAMS
        mask_params: 覆盖 mask 生成参数；None 时使用 config.MASK_PARAMS

    Returns:
        image_np: 预处理后的原图 [H,W,3], uint8
        edge_np:  边缘图 [H,W,3], uint8
        sketch_np: sketch 图 [H,W,3], uint8
        mask_np:  复杂 mask [H,W] 或 [H,W,3], uint8；若未启用复杂 mask，则为 None
    """
    modules = _get_modules()

    # 分辨率与开关的实际值
    effective_res = _effective_resolution(RESOLUTION if resolution is None else resolution, fallback=256)
    mask_flag = USE_COMPLEX_MASK if use_complex_mask is None else use_complex_mask

    # 1. 加载并 resize 原图
    image_np = _load_and_resize_image(image_path, effective_res)

    # 2. 提取 edge（带缓存）
    edge_np = _get_edge_image(image_np, image_path, modules, enable_cache=True)

    # 3. 生成 sketch
    make_sketch_from_image_or_edge = modules["make_sketch_from_image_or_edge"]

    base_sketch_params = modules["YZAPATCH_SKETCH_PARAMS"]
    override_sketch_params = SKETCH_PARAMS if sketch_params is None else sketch_params
    sp = _merge_params(base_sketch_params, override_sketch_params)

    seed = np.random.randint(0, 2**31 - 1)
    sketch_np = make_sketch_from_image_or_edge(
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

    # 4. 生成复杂 mask（可选）
    mask_np: Optional[np.ndarray] = None
    if mask_flag:
        from YZApatch.config import MASK_PARAMS as DEFAULT_MASK_PARAMS  # type: ignore
        from YZApatch.mask_generator import ComplexMaskGenerator  # type: ignore

        base_mask_params = DEFAULT_MASK_PARAMS
        override_mask_params = MASK_PARAMS if mask_params is None else mask_params
        mp = _merge_params(base_mask_params, override_mask_params)

        mask_generator = ComplexMaskGenerator(mp)
        mask_seed = np.random.randint(0, 2**31 - 1)
        mask_np = mask_generator.generate(edge_np, seed=int(mask_seed))

    return image_np, edge_np, sketch_np, mask_np

