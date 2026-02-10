#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SketchRefiner / SRN 在线数据预处理配置
=====================================

本文件只负责：
- 提供「覆盖参数占位」与「分辨率」等少量可调选项；
- 当这些参数保持默认值/为空时，应尽量回退到原有行为：
  - 分辨率默认 256；
  - sketch/mask 的详细参数从 diffusers 的 YZApatch.config 中读取默认值；
  - 是否使用复杂 mask 的选择交由训练脚本配置。

注意：
- 实际的 edge / sketch / mask 生成逻辑在 `YZA_patch/generator.py` 中实现；
- 你可以根据需要在这里填入想要覆盖的参数键值对。
"""

from typing import Dict, Any

#
# 分辨率相关
#

# 目标分辨率：
# - 默认设置为 256
# - 若与 SRN/SIN 的网络输入尺寸不同，调用方可以在外层再做一次 resize
RESOLUTION: int = 256

#
# mask 源选择
#

# 是否在训练中使用 ComplexMaskGenerator 生成的复杂 mask：
# - False：保持原始“自由涂抹”行为（由各自模块内部的 generate_stroke_mask 负责）
# - True：使用基于 edge 密度的复杂 mask（与 SDXL 训练时一致）
USE_COMPLEX_MASK: bool = False

#
# SIN 相关开关
#

# 是否在 SIN 训练中使用在线生成的 sketch / mask。
# - True：SIN_src.dataset.TrainDataset 会调用 YZA_patch.generator.generate_triplet
#         来生成 sketch/mask（与 SRN / YZApatch 行为一致）；
# - False：SIN 训练保持原始行为，从磁盘读取 sketch，并使用 generate_stroke_mask 生成 mask。
USE_ONLINE_SKETCH_FOR_SIN: bool = True

#
# sketch / mask 超参数覆盖
#

# 当下面两个字典为空时：
# - 实际使用的参数将回退到 diffusers/examples/controlnet/YZApatch/config.py 中
#   的 SKETCH_PARAMS / MASK_PARAMS 默认值；
# - 因此，保持空字典即可获得「与 SDXL 训练完全一致」的行为。

# 用于覆盖 sketch 生成参数的字典（留空则使用 YZApatch.config.SKETCH_PARAMS）
SKETCH_PARAMS: Dict[str, Any] = {
    # 示例（如需修改时，可取消注释并按需调整）：
    # "sigma_mean": 13.0,
    # "sigma_std": 2.6,
    # "spatial_smooth_sigma": 2.0,
    # "cp_sigma_mean": 2.1,
    # "cp_sigma_std": 0.4,
    # "cp_spatial_smooth": 1.5,
}

# 用于覆盖复杂 mask 生成参数的字典（留空则使用 YZApatch.config.MASK_PARAMS）
MASK_PARAMS: Dict[str, Any] = {
    # 示例（如需修改时，可取消注释并按需调整）：
    # "area_ratio_range": (0.2, 0.5),
    # "num_blocks_range": (5, 10),
}

