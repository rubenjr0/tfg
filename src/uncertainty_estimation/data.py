import random
from glob import glob
from os import listdir
from pathlib import Path

import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from matplotlib import colormaps

from uncertainty_estimation.utils import process_depth


class ImageDepthDataset(Dataset):
    def __init__(
        self,
        root: str = "data/",
        folders: list[str] = [],
        preprocess: bool = False,
    ):
        super().__init__()
        self.preprocess = preprocess
        if len(folders) > 0:
            all = [
                f
                for folder in folders
                for f in glob(f"{root}/{folder}/**/*final_preview/**", recursive=True)
            ]
        else:
            all = glob(f"{root}/**/*final_preview/**", recursive=True)
        self.paths = [
            (
                r,
                r.replace("final_preview", "geometry_hdf5").replace(
                    ".color.jpg", ".depth_meters.hdf5"
                ),
            )
            for r in all
            if ".color.jpg" in r
        ]
        self.transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float, scale=False),
                T.Resize((256, 256)),
            ]
        )

    def _aug(self, image, do_hflip: bool, do_rotate: bool, rotate_angle: float):
        if do_hflip:
            image = T.functional.hflip(image)
        if do_rotate:
            image = T.functional.rotate(image, rotate_angle)
        return image

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.paths[idx]
        image = Image.open(rgb_path).convert("RGB")
        depth = h5py.File(depth_path, "r")["dataset"][:]
        if np.isnan(image).any() or np.isnan(depth).any():
            return self.__getitem__((idx + 1) % len(self))

        image = self.transform(image)

        depth = self.transform(depth)
        depth_edges, depth_laplacian, _ = process_depth(depth.permute(1, 2, 0))
        depth_edges = self.transform(depth_edges)
        depth_laplacian = self.transform(depth_laplacian)

        output = {}

        if not self.preprocess:
            est = np.load(rgb_path.replace("color.jpg", "est.npy"))
            est = self.transform(est)
            est_edges, est_laplacian, _ = process_depth(est.permute(1, 2, 0))
            est_edges = self.transform(est_edges)
            est_laplacian = self.transform(est_laplacian)

            # Perform the same augmentations in the same way
            do_hflip = random.random() > 0.5
            do_rotate = random.random() > 0.5
            r_angle = random.uniform(-8, 8)
            image = self._aug(image, do_hflip, do_rotate, r_angle)
            depth = self._aug(depth, do_hflip, do_rotate, r_angle)
            depth_edges = self._aug(depth_edges, do_hflip, do_rotate, r_angle)
            depth_laplacian = self._aug(depth_laplacian, do_hflip, do_rotate, r_angle)
            est = self._aug(est, do_hflip, do_rotate, r_angle)
            est_edges = self._aug(est_edges, do_hflip, do_rotate, r_angle)
            est_laplacian = self._aug(est_laplacian, do_hflip, do_rotate, r_angle)

            output["est"] = est
            output["est_edges"] = est_edges
            output["est_laplacian"] = est_laplacian
        else:
            output["image_path"] = rgb_path

        output["image"] = image
        output["depth"] = depth
        output["depth_edges"] = depth_edges
        output["depth_laplacian"] = depth_laplacian

        return output


class UncertaintyDatamodule(LightningDataModule):
    def __init__(self, seed: int = 42, data_dir: str = "data/", batch_size: int = 16):
        super().__init__()
        self.batch_size = batch_size
        data_dir = Path(data_dir)
        self.train_dir = data_dir / "train"
        self.test_dir = data_dir / "test"
        train_folders = listdir(self.train_dir)
        self.train_folders, self.val_folders = train_test_split(
            train_folders, train_size=0.8, random_state=seed
        )
        self.train_data = ImageDepthDataset(
            root=self.train_dir, folders=self.train_folders
        )
        self.val_data = ImageDepthDataset(root=self.train_dir, folders=self.val_folders)
        self.test_data = ImageDepthDataset(root=self.test_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12,
            persistent_workers=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12,
            persistent_workers=True,
            drop_last=True,
        )


if __name__ == "__main__":
    ds = ImageDepthDataset()
    sample = ds[0]
    depth = sample["depth"]
    est = sample["est"]
    cmap = colormaps["turbo"]
    print("Depth:", depth.shape, depth.mean(), depth.std())
    print("Est:", est.shape, est.mean(), est.std())
    depth = cmap(depth[0])[..., :3]
    est = cmap(est[0])[..., :3]
    print("Colored Depth:", depth.shape, depth.mean(), depth.std())
    print("Colored Est:", est.shape, est.mean(), est.std())
