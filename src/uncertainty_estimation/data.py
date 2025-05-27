from glob import glob

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from uncertainty_estimation.utils import process_depth


class ImageDepthDataset(Dataset):
    def __init__(
        self,
        root: str,
        preprocess: bool = False,
        use_noise_augmentation: bool = True,
        noise_mean: float = 0.0,
        noise_std_range: tuple = (0.05, 0.2),
    ):
        super().__init__()
        self.preprocess = preprocess
        self.use_noise_augmentation = use_noise_augmentation
        self.noise_mean = noise_mean
        self.noise_std_range = noise_std_range
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

        output = {
            "image_path": rgb_path,
            "image": image,
            "depth": depth,
            "depth_edges": depth_edges,
            "depth_laplacian": depth_laplacian,
        }

        if not self.preprocess:
            est = np.load(rgb_path.replace("color.jpg", "est.npy"))
            est = self.transform(est)
            est_edges, est_laplacian, _ = process_depth(est.permute(1, 2, 0))
            est_edges = self.transform(est_edges)
            est_laplacian = self.transform(est_laplacian)

            noise_std = (
                torch.rand(1).item()
                * (self.noise_std_range[1] - self.noise_std_range[0])
                + self.noise_std_range[0]
            )
            noise = torch.randn_like(depth) * noise_std + self.noise_mean
            noisy_depth = depth + noise
            noisy_edges, noisy_laplacian, _ = process_depth(
                noisy_depth.permute(1, 2, 0)
            )
            noisy_edges = self.transform(noisy_edges)
            noisy_laplacian = self.transform(noisy_laplacian)

            output["est"] = est
            output["est_edges"] = est_edges
            output["est_laplacian"] = est_laplacian
            output["noise_std"] = noise_std
            output["noisy_depth"] = noisy_depth
            output["noisy_edges"] = noisy_edges
            output["noisy_laplacian"] = noisy_laplacian

        return output


if __name__ == "__main__":
    dataset = ImageDepthDataset(root="data/ai_001_001/images")
    entry = dataset[0]

    for k in entry.keys():
        d = entry[k].permute(1, 2, 0)
        print(f"{k:<20} {entry[k].shape} {entry[k].dtype}")
