import numpy as np

def load_timestamp_gt(gt_path):
    return np.load(gt_path, allow_pickle=True).item()


def save_timestamp_gt(gt_path, timestamps):
    np.save(gt_path, timestamps)


def create_missing_timestamp_gt(src_file, dst_file, miss_rate, seed=42):
    np.random.seed(seed)

    timestamps = load_timestamp_gt(src_file)

    # here, we randomly remove some timestamps
    for key in timestamps:
        timestamp_gt = timestamps[key]
        n = len(timestamp_gt)
        # n_miss = int(n * miss_rate)
        # not good, need to be probabilistic
        n_miss = np.random.binomial(n, miss_rate)
        idx = np.random.choice(n, n_miss, replace=False)
        timestamp_gt = np.delete(timestamp_gt, idx)
        timestamps[key] = timestamp_gt

    save_timestamp_gt(dst_file, timestamps)


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Create missing timestamp ground truth")
    parser.add_argument("--src_file", type=str, help="source file")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--dst_file", type=str, help="destination file")
    parser.add_argument("--miss_rate", type=float,
                        help="missing rate", default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    # assert that only src_file or dataset is provided, and at least one of them is provided
    assert (args.src_file is not None) or (args.dataset is not None)
    assert (args.src_file is None) or (args.dataset is None)
    if args.src_file is None:
        args.src_file = f"data/{args.dataset}/groundTruth/{
            args.dataset}_annotation_all.npy"

    if args.dst_file is None:
        args.dst_file = f"data/{args.dataset}/groundTruth/{
            args.dataset}_annotation_all_{args.miss_rate}_missing.npy"

    return args


if __name__ == "__main__":

    args = get_args()

    src_file = args.src_file
    dst_file = args.dst_file
    miss_rate = args.miss_rate
    seed = args.seed

    create_missing_timestamp_gt(src_file, dst_file, miss_rate, seed)
