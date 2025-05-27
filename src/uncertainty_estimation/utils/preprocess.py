import numpy as np

from uncertainty_estimation.utils import Pipeline
from uncertainty_estimation.data import ImageDepthDataset
from tqdm import tqdm


def preprocess():
    pipeline = Pipeline(None, None)
    dataset = ImageDepthDataset(root="data/", preprocess=True)
    for k in tqdm(range(len(dataset))):
        entry = dataset[k]
        rgb_path = entry["image_path"]
        image = entry["image"]
        estimated = pipeline.estimate_depth(image)
        estimated = estimated.cpu().numpy()
        path = rgb_path.replace("color.jpg", "est.npy")
        np.save(path, estimated)
