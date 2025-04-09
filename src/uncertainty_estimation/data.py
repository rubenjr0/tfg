import glob
import os
import pathlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from uncertainty_estimation.utils import Pipeline, Sensor


class ImageDepthDataset(Dataset):
    def __init__(self, root: str, sensor: Sensor):
        super().__init__()
        root = pathlib.Path(root)
        self.sensor = sensor
        all_files = glob.glob(os.path.join(root, "**/*.png"), recursive=True)
        self.image_pairs = [
            (
                f,
                f.replace(".png", "_depth.npy"),
                f.replace(".png", "_est.npy"),
            )
            for f in all_files
        ]
        self.transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float, scale=False),
                T.Resize((128, 128)),
            ]
        )

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image_path, depth_path, est_path = self.image_pairs[idx]
        image = Image.open(image_path).convert("RGB")
        depth = np.load(depth_path)

        est = torch.from_numpy(np.load(est_path))
        image = self.transform(image)
        depth = self.transform(depth)
        depth_np = depth.numpy().transpose(1, 2, 0)
        depth, depth_edges, depth_laplacian, mask = Pipeline.process_depth(depth_np)
        depth_var = self.sensor.compute_uncertainty(depth, depth_edges)
        depth_laplacian = torch.from_numpy(depth_laplacian).permute(2, 0, 1)
        depth = torch.from_numpy(depth).permute(2, 0, 1)
        depth_var = torch.from_numpy(depth_var).permute(2, 0, 1)
        depth_edges = torch.from_numpy(depth_edges).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        est = self.transform(est)
        return {
            "image": image,
            "depth": depth,
            "depth_var": depth_var,
            "depth_edges": depth_edges,
            "depth_laplacian": depth_laplacian,
            "mask": mask,
            "est": est,
        }


if __name__ == "__main__":
    sensor = Sensor(0.3, 20, (0.0008, 0.0016, 0.0018))
    dataset = ImageDepthDataset(root="data/val/", sensor=sensor)
    batch = dataset[0]

    for k in batch.keys():
        print(f"{k:<20} {batch[k].shape} {batch[k].dtype}")
