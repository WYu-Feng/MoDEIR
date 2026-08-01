import os
import glob
import random
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class AlignedDataset(data.Dataset):
    """
    Multi-task paired dataset (derain/dehaze/deblur/lowlight/denoise)
    - Pair by basename intersection (robust)
    - train: resize-if-small + random crop
    - val/test: resize-if-small + center crop
    - output: (condition, gt) in [-1, 1], to match SD/LDM VAE expectation
    """
    def __init__(
        self,
        root="./datasets",
        mode="test",
        train_size=448,
        augment_flip=True,
        target_per_task=10000,
        use_tasks=("derain", "dehaze", "deblur", "light", "noise"),
    ):
        super().__init__()
        self.root = root
        self.mode = mode
        self.train_size = train_size
        self.augment_flip = augment_flip
        self.target_per_task = target_per_task
        self.use_tasks = set(use_tasks)

        self.to_tensor = transforms.ToTensor()
        self.pairs = []

        # ------- build pairs per task -------
        A, B = self.read_derain()
        pairs = self.pair_by_basename(A, B)
        pairs = self.balance_pairs(pairs, target_per_task)
        self.pairs += [(a, b, "derain") for a, b in pairs]
        print(f"[DATA] Derain: A={len(A)} B={len(B)} paired={len(pairs)}")

        A, B = self.read_dehaze()
        pairs = self.pair_by_basename(A, B)
        pairs = self.balance_pairs(pairs, target_per_task)
        self.pairs += [(a, b, "dehaze") for a, b in pairs]
        print(f"[DATA] Dehaze: A={len(A)} B={len(B)} paired={len(pairs)}")

        A, B = self.read_deblur()
        pairs = self.pair_by_basename(A, B)
        pairs = self.balance_pairs(pairs, target_per_task)
        self.pairs += [(a, b, "deblur") for a, b in pairs]
        print(f"[DATA] Deblur: A={len(A)} B={len(B)} paired={len(pairs)}")

        A, B = self.read_light()
        pairs = self.pair_by_basename(A, B)
        pairs = self.balance_pairs(pairs, target_per_task)
        self.pairs += [(a, b, "light") for a, b in pairs]
        print(f"[DATA] Light:  A={len(A)} B={len(B)} paired={len(pairs)}")

        A, B = self.read_noise()
        pairs = self.pair_by_basename(A, B)
        pairs = self.balance_pairs(pairs, target_per_task)
        self.pairs += [(a, b, "noise") for a, b in pairs]
        print(f"[DATA] Noise:  A={len(A)} B={len(B)} paired={len(pairs)}")

        print(f"[DATA] Total paired samples: {len(self.pairs)}")

    # -------------------- pairing & balancing --------------------
    @staticmethod
    def pair_by_basename(A_paths, B_paths):
        """
        Return list of (A_path, B_path) paired by identical basename.
        Unmatched files are dropped.
        """
        A_map = {os.path.basename(p): p for p in A_paths}
        B_map = {os.path.basename(p): p for p in B_paths}
        keys = sorted(list(set(A_map.keys()) & set(B_map.keys())))
        return [(A_map[k], B_map[k]) for k in keys]

    @staticmethod
    def balance_pairs(pairs, target):
        """
        Balance a list of pairs to target length by downsample or repeat.
        """
        n = len(pairs)
        if n == 0:
            return []
        if n > target:
            idx = random.sample(range(n), target)
            return [pairs[i] for i in idx]
        else:
            k = target // n
            r = target % n
            out = pairs * k + random.choices(pairs, k=r)
            return out

    # -------------------- IO & transforms (match PairedGoProCropDataset) --------------------
    def _load_pair(self, deg_path, gt_path):
        condition = Image.open(deg_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        # if size mismatch, center-crop to common min size (safe)
        w, h = condition.size
        w2, h2 = gt.size
        if (w, h) != (w2, h2):
            ww, hh = min(w, w2), min(h, h2)
            condition = TF.center_crop(condition, [hh, ww])
            gt = TF.center_crop(gt, [hh, ww])
            w, h = condition.size

        ts = self.train_size

        if self.mode == "train":
            # resize-if-small
            if h < ts or w < ts:
                resi = transforms.Resize([ts, ts], interpolation=InterpolationMode.BICUBIC)
                condition = resi(condition)
                gt = resi(gt)
                w, h = condition.size

            # random crop
            crop_x = random.randint(0, w - ts)
            crop_y = random.randint(0, h - ts)
            condition = TF.crop(condition, crop_y, crop_x, ts, ts)
            gt = TF.crop(gt, crop_y, crop_x, ts, ts)

        else:
            # val/test: ensure not smaller than ts then center crop
            if h < ts or w < ts:
                resi = transforms.Resize([ts, ts], interpolation=InterpolationMode.BICUBIC)
                condition = resi(condition)
                gt = resi(gt)
            condition = TF.center_crop(condition, [ts, ts])
            gt = TF.center_crop(gt, [ts, ts])

        condition = self.to_tensor(condition)  # [0,1]
        gt = self.to_tensor(gt)

        # to [-1,1] for SD/LDM VAE
        condition = condition * 2.0 - 1.0
        gt = gt * 2.0 - 1.0
        return condition, gt

    def __getitem__(self, idx):
        deg_p, gt_p, task = self.pairs[idx]
        # 与 PairedGoProCropDataset 保持一致：返回 (condition, gt)
        # 如果你还想要 task label / path，可以改成 return condition, gt, task
        return self._load_pair(deg_p, gt_p)

    def __len__(self):
        return len(self.pairs)

    # -------------------- your original readers (kept, but root-aware) --------------------
    def read_deblur(self):
        A = glob.glob(os.path.join(self.root, "Image deblurring", "GoPro", "train", "input", "*.png")) + \
            glob.glob(os.path.join(self.root, "Image deblurring", "GoPro", "test", "input", "*.png"))
        B = glob.glob(os.path.join(self.root, "Image deblurring", "GoPro", "train", "target", "*.png")) + \
            glob.glob(os.path.join(self.root, "Image deblurring", "GoPro", "test", "target", "*.png"))
        return A, B

    def read_derain(self):
        A = glob.glob(os.path.join(self.root, "Image deraining", "Rain100L", "input", "*.png")) + \
            glob.glob(os.path.join(self.root, "Image deraining", "RainTrainL", "input", "*.png"))
        B = glob.glob(os.path.join(self.root, "Image deraining", "Rain100L", "target", "*.png")) + \
            glob.glob(os.path.join(self.root, "Image deraining", "RainTrainL", "target", "*.png"))
        return A, B

    def read_dehaze(self):
        A = glob.glob(os.path.join(self.root, "Image dehazing", "SOTS", "outdoor", "hazy", "*.png"))
        B = glob.glob(os.path.join(self.root, "Image dehazing", "SOTS", "outdoor", "gt", "*.png"))
        return A, B

    def read_light(self):
        A = glob.glob(os.path.join(self.root, "Low-light enhancement", "LOL", "train", "low", "*.png")) + \
            glob.glob(os.path.join(self.root, "Low-light enhancement", "LOL", "eval", "low", "*.png"))
        B = glob.glob(os.path.join(self.root, "Low-light enhancement", "LOL", "train", "high", "*.png")) + \
            glob.glob(os.path.join(self.root, "Low-light enhancement", "LOL", "eval", "high", "*.png"))
        return A, B

    def read_noise(self):
        A = glob.glob(os.path.join(self.root, "Image denoise", "Urban100", "2", "input", "*.png")) + \
            glob.glob(os.path.join(self.root, "Image denoise", "Urban100", "4", "input", "*.png")) + \
            glob.glob(os.path.join(self.root, "Image denoise", "BSD68", "noisy25", "*.png"))
        B = glob.glob(os.path.join(self.root, "Image denoise", "Urban100", "2", "target", "*.png")) + \
            glob.glob(os.path.join(self.root, "Image denoise", "Urban100", "4", "target", "*.png")) + \
            glob.glob(os.path.join(self.root, "Image denoise", "BSD68", "original", "*.png"))
        return A, B
