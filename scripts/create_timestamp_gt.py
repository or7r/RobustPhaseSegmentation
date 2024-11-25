import numpy as np
from skimage import measure
import os


def read_gt(gt_path):
    with open(gt_path, "r") as f:
        lines = f.read()
    lines = lines.split("\n")
    lines = lines[:-1]
    lines = np.array(lines)
    # lines = lines.astype(np.int)
    return lines


def make_timestamp_gt(root_gt_path, save_path, seed=42):
    np.random.seed(seed)
    timestamps = {}

    for gt_path in os.listdir(root_gt_path):
        if not gt_path.endswith(".txt"):
            continue
        timestamp_gt = []
        gt_path = os.path.join(root_gt_path, gt_path)
        gt = read_gt(gt_path)

        # get unique action phases
        actions = np.unique(gt)
        for act in actions:
            labels, num = measure.label(gt == act, return_num=True)
            # select one random frame from each action phase
            # to represent the action phase
            for i in range(1, num + 1):
                # get all frames of the action phase
                frames = np.argwhere(labels == i)
                frames = frames.flatten()
                # randomly select one frame
                frame = np.random.choice(frames)
                timestamp_gt.append(frame)
        timestamp_gt = np.array(timestamp_gt)
        # sort the timestamps
        timestamp_gt = np.sort(timestamp_gt)

        timestamps[os.path.basename(gt_path)] = timestamp_gt

    np.save(save_path, timestamps)


if __name__ == "__main__":

    dataset = "MultiBypass140"
    gt_path = f"data/{dataset}/groundTruth"
    save_path = os.path.join(gt_path, f"{dataset}_annotation_all.npy")
    make_timestamp_gt(gt_path, save_path)
