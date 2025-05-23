import numpy as np

from uncertainty_estimation.utils import Pipeline

if __name__ == "__main__":
    from uncertainty_estimation.data import ImageDepthDataset
    from tqdm import tqdm

    pipeline = Pipeline(None, None)
    dataset = ImageDepthDataset(root="data/ai_001_001/images")
    for k in tqdm(range(len(dataset))):
        entry = dataset[k]
        rgb_path = entry["path"]
        image = entry["image"]
        estimated = pipeline.estimate_depth(image)
        estimated = estimated.cpu().numpy()
        path = rgb_path.replace("color.jpg", "est.npy")
        np.save(path, estimated)
