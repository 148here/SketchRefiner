import os
import os.path as osp
import sys
import random
import numpy as np
import cv2

from SRN_src.utils import RandomDeformSketch, generate_stroke_mask

# 在线生成 edge / sketch / mask 的工具与配置
try:
    from YZA_patch.config import (
        RESOLUTION as YZA_RESOLUTION,
        USE_COMPLEX_MASK,
        SKETCH_PARAMS,
        MASK_PARAMS,
        EXTERNAL_EDGE_POLARITY,
        MODEL_EDGE_POLARITY,
    )
    from YZA_patch.generator import generate_triplet
except ImportError:
    # 若导入失败，仍然允许使用旧的离线数据流程
    YZA_RESOLUTION = None
    USE_COMPLEX_MASK = False
    SKETCH_PARAMS = {}
    MASK_PARAMS = {}
    EXTERNAL_EDGE_POLARITY = "black_on_white"
    MODEL_EDGE_POLARITY = "white_on_black"
    generate_triplet = None

import torch
from torch.utils.data import DataLoader


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _normalize_gray01(image):
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    array = array.astype(np.float32)
    if array.size and array.max() > 1.0:
        array = array / 255.0
    return array


def _binarize_edge_like(image, threshold):
    binary = (_normalize_gray01(image) > float(threshold)).astype(np.float32)
    return np.expand_dims(binary, axis=-1)


def _convert_line_polarity(binary, source_polarity, target_polarity):
    source = str(source_polarity or "black_on_white").strip().lower()
    target = str(target_polarity or "white_on_black").strip().lower()
    valid = {"black_on_white", "white_on_black"}
    if source not in valid:
        raise ValueError("Unsupported source edge polarity: %s" % source_polarity)
    if target not in valid:
        raise ValueError("Unsupported target edge polarity: %s" % target_polarity)
    if source == target:
        return binary.astype(np.float32)
    return (1.0 - binary).astype(np.float32)


def _external_edge_to_model(image, threshold):
    binary = _binarize_edge_like(image, threshold)
    return _convert_line_polarity(binary, EXTERNAL_EDGE_POLARITY, MODEL_EDGE_POLARITY)


# train dataset
class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, configs):
        super(TrainDataset, self).__init__()

        self.configs = configs

        # 目标分辨率：优先使用 YZA_patch.config 中的 RESOLUTION，其次使用 configs.size
        self.target_size = int(YZA_RESOLUTION) if YZA_RESOLUTION not in (None, 0) else int(self.configs.size)

        # deforming algorithm（仅在无法使用在线生成器时回退使用）
        self.deform_func = RandomDeformSketch(input_size=self.target_size)
        self.max_move = random.randint(configs.max_move_lower_bound, configs.max_move_upper_bound)

        self.image_flist = sorted(self.get_files_from_path(self.configs.images))
        if len(self.image_flist) == 0:
            raise ValueError("No training images found from --images: %s" % self.configs.images)


    def __len__(self):

        return len(self.image_flist)


    def __getitem__(self, index):
        retry_limit = int(getattr(self.configs, "sample_retry_limit", 0) or 0)
        if generate_triplet is None or retry_limit <= 0:
            return self._load_item(index)

        last_error = None
        dataset_size = len(self.image_flist)
        for attempt in range(max(1, retry_limit)):
            candidate_index = index if attempt == 0 else random.randint(0, dataset_size - 1)
            try:
                return self._load_item(candidate_index)
            except Exception as e:
                last_error = e
                image_path = self.image_flist[candidate_index]
                print(
                    "[TrainDataset] skip failed sample attempt=%d/%d path=%s error=%s"
                    % (attempt + 1, retry_limit, image_path, repr(e))
                )
                sys.stdout.flush()

        raise RuntimeError(
            "Failed to load an online-sketch sample after %d attempts. Last error: %r"
            % (retry_limit, last_error)
        )


    def _load_item(self, index):

        data = {}
        image_path = self.image_flist[index]

        # ============================================================
        # 优先使用在线生成器（与 SDXL / YZApatch 一致的流程）
        # ============================================================
        if generate_triplet is not None:
            img_np, edge_np, sketch_np, mask_np = generate_triplet(
                image_path=image_path,
                resolution=self.target_size,
                use_complex_mask=USE_COMPLEX_MASK,
                sketch_params=SKETCH_PARAMS,
                mask_params=MASK_PARAMS,
            )

            # 若 SRN 的 configs.size 与 target_size 不一致，则在此处做一次额外 resize
            if self.configs.size != self.target_size:
                img_np = cv2.resize(img_np, (self.configs.size, self.configs.size), interpolation=cv2.INTER_LINEAR)
                edge_np = cv2.resize(edge_np, (self.configs.size, self.configs.size), interpolation=cv2.INTER_LINEAR)
                sketch_np = cv2.resize(sketch_np, (self.configs.size, self.configs.size), interpolation=cv2.INTER_LINEAR)
                if mask_np is not None:
                    mask_np = cv2.resize(mask_np, (self.configs.size, self.configs.size), interpolation=cv2.INTER_NEAREST)
        else:
            # 回退：保持旧的「磁盘固定 edge + RandomDeformSketch」行为
            img_np = cv2.imread(image_path)
            filename = osp.basename(image_path)

            if filename.split('.')[1] == "JPEG" or filename.split('.')[1] == "jpg":
                filename = filename.split('.')[0] + '.png'

            edge_np = cv2.imread(osp.join(self.configs.edges_prefix, filename))
            sketch_np = None
            mask_np = None

            img_np = cv2.resize(img_np, (self.configs.size, self.configs.size), interpolation=cv2.INTER_NEAREST)
            edge_np = cv2.resize(edge_np, (self.configs.size, self.configs.size), interpolation=cv2.INTER_NEAREST)

        # 归一化到 [0,1]
        data['image'] = img_np.astype(np.float32) / 255.0

        # mask：
        # - 若 USE_COMPLEX_MASK=True 且在线生成器返回 mask_np，则直接使用；
        # - 否则使用 SRN 原有的 generate_stroke_mask（自由涂抹）。
        if USE_COMPLEX_MASK and mask_np is not None:
            if mask_np.ndim == 2:
                mask_np_expanded = np.expand_dims(mask_np, axis=-1)
            else:
                mask_np_expanded = mask_np
            mask_np_expanded = mask_np_expanded.astype(np.float32) / 255.0
            if mask_np_expanded.shape[:2] != (self.configs.size, self.configs.size):
                mask_np_expanded = cv2.resize(
                    mask_np_expanded,
                    (self.configs.size, self.configs.size),
                    interpolation=cv2.INTER_NEAREST,
                )
            data['mask'] = mask_np_expanded
        else:
            data['mask'] = generate_stroke_mask(im_size=[self.configs.size, self.configs.size])

        # resize 到网络输入尺寸（若上面已是 configs.size，这一步等价于 no-op）
        if data['image'].shape[0:2] != (self.configs.size, self.configs.size):
            data['image'] = cv2.resize(
                data['image'],
                (self.configs.size, self.configs.size),
                interpolation=cv2.INTER_LINEAR,
            )
        if edge_np.shape[0:2] != (self.configs.size, self.configs.size):
            edge_np = cv2.resize(
                edge_np,
                (self.configs.size, self.configs.size),
                interpolation=cv2.INTER_LINEAR,
            )

        # 二值化 edge（与原始实现保持一致）
        thresh = random.uniform(0.65, 0.75)
        data['edge'] = _external_edge_to_model(edge_np, thresh)

        # [H, W, C] -> [C, H, W]
        data['image'] = torch.from_numpy(data['image']).permute(2, 0, 1).contiguous()
        data['mask'] = torch.from_numpy(data['mask'].astype(np.float32)).permute(2, 0, 1).contiguous()
        data['edge'] = torch.from_numpy(data['edge'].astype(np.float32)).permute(2, 0, 1).contiguous()

        # sketch：
        # - 在线模式：直接使用 generator 返回的 sketch_np；
        # - 回退模式：使用 RandomDeformSketch(edge) 生成。
        if sketch_np is not None:
            sketch_model = _external_edge_to_model(sketch_np, 0.5)
            sketch_tensor = torch.from_numpy(sketch_model).permute(2, 0, 1).contiguous()
        else:
            sketch_tensor = self.deform_func(data['edge'].unsqueeze(0), self.max_move).squeeze(0)

        # 压到单通道
        if sketch_tensor.size(0) > 1:
            data['sketch'] = torch.sum(sketch_tensor / 3.0, dim=0, keepdim=True)
        else:
            data['sketch'] = sketch_tensor
        if data['edge'].size(0) > 1:
            data['edge'] = torch.sum(data['edge'] / 3.0, dim=0, keepdim=True)

        # 返回 data：image, mask, sketch, edge
        data['sketch'] = data['sketch'].detach()
        return data


    def get_files_from_txt(self, path):

        file_list = []
        f = open(path)
        for line in f.readlines():
            line = line.strip("\n")
            file_list.append(line)
            sys.stdout.flush()
        f.close()

        return file_list


    def get_files_from_path(self, path):
        """Read image folders. Comma/semicolon separated roots are supported."""
        if not path or not path.strip():
            return []
        paths = [p.strip() for p in path.replace(';', ',').split(',') if p.strip()]
        ret = []
        scan_mode = getattr(self.configs, "image_scan_mode", "recursive")
        for p in paths:
            if osp.isfile(p):
                if self._is_image_file(p):
                    ret.append(p)
            elif osp.isdir(p):
                if scan_mode == "stage3":
                    ret.extend(self._scan_stage3_root(p))
                else:
                    ret.extend(self._scan_image_tree(p))
        return ret


    def _is_image_file(self, path):
        return osp.splitext(path)[1].lower() in IMAGE_EXTENSIONS


    def _scan_image_tree(self, root_dir):
        ret = []
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                path = osp.join(root, f)
                if self._is_image_file(path):
                    ret.append(path)
        return ret


    def _scan_stage3_root(self, dataset_root):
        split = str(getattr(self.configs, "stage3_split", "train") or "train")
        split_root = osp.join(dataset_root, split)
        if not osp.isdir(split_root):
            split_root = dataset_root

        image_dirs = []
        for root, dirs, files in os.walk(split_root):
            if osp.basename(root).lower() == "images":
                image_dirs.append(root)

        ret = []
        for image_dir in image_dirs:
            ret.extend(self._scan_image_tree(image_dir))

        if not ret:
            ret.extend(self._scan_image_tree(split_root))

        return ret


# validation dataset
class ValDataset(torch.utils.data.Dataset):
    
    def __init__(self, configs):
        super(ValDataset, self).__init__()

        self.configs = configs
        self.size = int(self.configs.size)

        # 如果提供了 val_root_dir，则使用「母目录 + *_edge/_sketch/_mask」的读取逻辑
        self.val_root_dir = getattr(self.configs, "val_root_dir", "")
        if self.val_root_dir:
            self.val_root_dir = osp.abspath(self.val_root_dir)
            self.samples = self._build_samples_from_root(self.val_root_dir)
            self.use_root_dir = True
        else:
            # 兼容旧版：使用 txt 列表 + 独立文件夹
            self.deform_func = RandomDeformSketch(input_size=configs.size)
            self.max_move = random.randint(30, 100)

            self.image_flist = sorted(self.get_files_from_txt(self.configs.images_val))
            self.mask_flist = sorted(self.get_files_from_txt(self.configs.masks_val))
            self.use_root_dir = False


    def __len__(self):

        if getattr(self, "use_root_dir", False):
            return len(self.samples)
        return len(self.image_flist)


    def __getitem__(self, index):

        data = {}
        # 新格式：从母目录中读取原图 + *_edge/_sketch/_mask
        if getattr(self, "use_root_dir", False):
            sample = self.samples[index]
            img_np = cv2.imread(sample['image'])
            edge_np = cv2.imread(sample['edge'])
            sketch_np = cv2.imread(sample['sketch'])
            mask_np = cv2.imread(sample['mask'])

            filename = osp.basename(sample['image'])

            # 归一化
            img_np = img_np.astype(np.float32) / 255.0
            edge_np = edge_np.astype(np.float32) / 255.0
            sketch_np = sketch_np.astype(np.float32) / 255.0
            mask_np = mask_np.astype(np.float32) / 255.0

            # resize
            img_np = cv2.resize(img_np, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            edge_np = cv2.resize(edge_np, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            sketch_np = cv2.resize(sketch_np, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            mask_np = cv2.resize(mask_np, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

            # 二值化 edge / mask（sketch 保持灰度即可）
            thresh = random.uniform(0.65, 0.75)
            _, edge_np = cv2.threshold(edge_np, thresh=thresh, maxval=1.0, type=cv2.THRESH_BINARY)
            _, mask_np = cv2.threshold(mask_np, thresh=0.5, maxval=1.0, type=cv2.THRESH_BINARY)

            # 转 tensor
            data['image'] = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
            data['edge'] = torch.from_numpy(edge_np).permute(2, 0, 1).contiguous()
            data['sketch'] = torch.from_numpy(sketch_np).permute(2, 0, 1).contiguous()
            data['mask'] = torch.from_numpy(mask_np).permute(2, 0, 1).contiguous()

            # 压到单通道
            data['edge'] = torch.sum(data['edge'] / 3.0, dim=0, keepdim=True)
            data['sketch'] = torch.sum(data['sketch'] / 3.0, dim=0, keepdim=True)
            data['mask'] = torch.sum(data['mask'] / 3.0, dim=0, keepdim=True)

            data['filename'] = filename
            return data

        # 旧格式：保持原有 txt + 独立文件夹 + RandomDeformSketch 行为
        data['image'] = cv2.imread(self.image_flist[index])

        filename = osp.basename(self.image_flist[index])

        if filename.split('.')[1] == "JPEG":
            filename = filename.split('.')[0] + '.png'

        data['edge'] = cv2.imread(osp.join(self.configs.edges_prefix_val, filename))

        # generate free-form mask
        data['mask'] = cv2.imread(self.mask_flist[index])

        # normalize
        data['image'] = data['image'] / 255.
        data['mask'] = data['mask'] / 255.
        data['edge'] = data['edge'] / 255.

        # resize
        data['image'] = cv2.resize(data['image'], (self.configs.size, self.configs.size))
        data['mask'] = cv2.resize(data['mask'], (self.configs.size, self.configs.size), interpolation=cv2.INTER_NEAREST)
        data['edge'] = cv2.resize(data['edge'], (self.configs.size, self.configs.size), interpolation=cv2.INTER_NEAREST)

        # binarize
        thresh = random.uniform(0.65, 0.75)
        _, data['mask'] = cv2.threshold(data['mask'], thresh=0.5, maxval=1.0, type=cv2.THRESH_BINARY)
        _, data['edge'] = cv2.threshold(data['edge'], thresh=thresh, maxval=1.0, type=cv2.THRESH_BINARY)

        # to tensor
        data['image'] = torch.from_numpy(data['image'].astype(np.float32)).permute(2,0,1).contiguous()
        data['mask'] = torch.from_numpy(data['mask'].astype(np.float32)).permute(2,0,1).contiguous()
        data['edge'] = torch.from_numpy(data['edge'].astype(np.float32)).permute(2,0,1).contiguous()

        # compress RGB channels to 1 ([C=1, H, W])
        data['mask'] = torch.sum(data['mask'] / 3, dim=0, keepdim=True)
        data['edge'] = torch.sum(data['edge'] / 3, dim=0, keepdim=True)

        # generate deform sketches
        data['sketch'] = self.deform_func(data['edge'].unsqueeze(0), self.max_move).squeeze(0).detach()

        data['filename'] = filename
        return data


    def get_files_from_txt(self, path):

        file_list = []
        f = open(path)
        for line in f.readlines():
            line = line.strip("\n")
            file_list.append(line)
            sys.stdout.flush()
        f.close()


        return file_list


    def get_files(self, path):

        # read a folder, return the complete path
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))

        return ret


    def _build_samples_from_root(self, root_dir):
        """
        从一个母目录中递归查找原图，并匹配同目录下的 *_edge / *_sketch / *_mask。
        """
        exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        samples = []

        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                path = osp.join(dirpath, name)
                ext = osp.splitext(name)[1].lower()
                stem = osp.splitext(name)[0]

                # 跳过已经是派生文件的样本
                if not ext or ext not in exts:
                    continue
                if stem.endswith('_edge') or stem.endswith('_sketch') or stem.endswith('_mask'):
                    continue

                base_stem = stem
                parent = dirpath

                def _find_with_suffix(suffix):
                    candidate = osp.join(parent, base_stem + suffix + ext)
                    return candidate if osp.isfile(candidate) else None

                edge_path = _find_with_suffix('_edge')
                sketch_path = _find_with_suffix('_sketch')
                mask_path = _find_with_suffix('_mask')

                if edge_path is None or sketch_path is None or mask_path is None:
                    continue

                samples.append({
                    'image': path,
                    'edge': edge_path,
                    'sketch': sketch_path,
                    'mask': mask_path,
                })

        if len(samples) == 0:
            raise ValueError(f"No valid validation samples found in root_dir: {root_dir}")

        return samples

if __name__ == '__main__':
    pass
