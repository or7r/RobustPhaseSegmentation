import numpy as np


def load(path):
    return np.load(path, allow_pickle=True).item()

def save(path, labels):
    np.save(path, labels, allow_pickle=True)

def select_half(labels):
    # drop half of the keys
    keys = list(labels.keys())
    keys = keys[:len(keys)//2]
    labels = {k: labels[k] for k in keys}
    return labels

def split2files(orig_path):
    labels = load(orig_path)
    folder_name = os.path.dirname(orig_path)
    base_name = os.path.basename(orig_path)

    # filter by all keys that contain 'BBP'
    bbp_labels = {k: v for k, v in labels.items() if 'BBP' in k}
    bbp_labels_path = os.path.join(folder_name, base_name + '_bbp.npy')
    # bbp_labels = select_half(bbp_labels)
    save(bbp_labels_path, bbp_labels)
    
    sbp_labels = {k: v for k, v in labels.items() if 'SBP' in k}
    sbp_labels_path = os.path.join(folder_name, base_name + '_sbp.npy')
    # sbp_labels = select_half(sbp_labels)
    save(sbp_labels_path, sbp_labels)


    half_labels = select_half(labels)
    half_labels_path = os.path.join(folder_name, base_name + '_half.npy')
    save(half_labels_path, half_labels)

if __name__ == '__main__':
    import os
    import sys

    if len(sys.argv) != 2:
        print('Usage: python split_labels.py <path>')
        sys.exit(1)

    orig_path = sys.argv[1]
    split2files(orig_path)