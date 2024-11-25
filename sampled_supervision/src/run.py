import os
import numpy as np
import torch
import random
import argparse

from label_handler import LabelCreator

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def create_action_dict(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    return actions_dict

def make_labels(dataset, splits_path, save_path, data_path="data/", seed=42, region_num=10, label_type="uniform", artificial_num=0):

    assert dataset in ["cholec80", "MultiBypass140"]
    if seed is not None:
        set_seed(seed)

    sample_rate = 1
    vid_list_file = splits_path + f"{dataset}_all.bundle"
    gt_path = data_path + f"{dataset}/groundTruth/"
    mapping_file = data_path + f"{dataset}/mapping.txt"

    actions_dict = create_action_dict(mapping_file)

    label_creator = LabelCreator(vid_list_file, gt_path, actions_dict, sample_rate)
    if label_type == "uniform":
        sampled_labels = label_creator.get_sampled_labels(region_num)
    elif label_type == "fixed":
        sampled_labels = label_creator.get_fixed_labels(region_num)
    elif label_type == "random":
        sampled_labels = label_creator.get_random_labels(region_num)
    else:
        raise ValueError(f"Unknown label type: {label_type}")
    
    if artificial_num > 0:
        sampled_labels = label_creator.add_artificial(sampled_labels, artificial_num)

    label_creator.save_samples(sampled_labels, save_path)


def arg_parse():
    parser = argparse.ArgumentParser(description='Sample Supervision')
    parser.add_argument('--dataset', type=str, default="cholec80", help='cholec80 or MultiBypass140', choices=['cholec80', 
                                                                                                               'MultiBypass140'])
    parser.add_argument('--region_num', type=int, default=10, help='number of regions to sample')
    parser.add_argument('--sample_width', type=int, default=1, help='number of frames to sample from each region')
    parser.add_argument('--splits_path', type=str, default="sampled_supervision/splits/", help='path to splits')
    parser.add_argument('--save_path', type=str, default=None, help='path to save labels')
    parser.add_argument('--data_path', type=str, default="data/", help='path to data')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--label_type', type=str, default="uniform", help='uniform, fixed, random')
    parser.add_argument('--artificial_num', type=int, default=0, help='add artificial labels to the dataset')
    args = parser.parse_args()

    assert args.sample_width > 0 and args.sample_width % 2, "Sample width must be greater than 0, and odd."

    if args.save_path is None:
        save_path = os.path.join("sampled_supervision/labels")
        file_name = f"{args.dataset}_{args.region_num}regions"
        if args.label_type != "uniform":
            file_name += f"_type{args.label_type}"
        if args.artificial_num > 0:
            file_name += f"_artificial{args.artificial_num}"
        if args.sample_width > 1:
            file_name += f"_width{args.sample_width}"
        file_name += ".npy"

        args.save_path = os.path.join(save_path, file_name)
        

    # if args.save_path is None and args.label_type == "uniform":
    #     args.save_path = f"sampled_supervision/labels/{args.dataset}_{args.region_num}regions.npy"
    # elif args.save_path is None:
    #     args.save_path = f"sampled_supervision/labels/{args.dataset}_{args.region_num}regions_{args.label_type}.npy"
    
    
    return args

    




if __name__ == "__main__":

    args = arg_parse()

    make_labels(args.dataset,
                args.splits_path,
                args.save_path, 
                seed=args.seed,
                region_num=args.region_num,
                label_type=args.label_type,
                artificial_num=args.artificial_num)


