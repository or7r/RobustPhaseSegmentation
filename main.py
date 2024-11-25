#!/usr/bin/python3.6

# IMPORTANT: do not touch following import
import gtn


import torch
from torch.utils.tensorboard import SummaryWriter
from model import Trainer
from batch_gen import BatchGenerator
import argparse
import random
import time
import os
import sys
import re
from eval import evaluate
import numpy as np
import wandb
import yaml
# import json
import ast

from loguru import logger

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# comment out seed to train the model
# seed = 1538574472


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='train',
                        help='two options: train or predict')
    parser.add_argument('--dataset', default="breakfast", help='dataset name: cholec80 or MultiBypass140',
                        choices=['cholec80', 'MultiBypass140'])
    parser.add_argument('--split', default='1',
                        help="split number or 'all' for all splits")
    parser.add_argument('--exp_name', default='test', help='experiment name')
    parser.add_argument('--ctc_weight', default=100,
                        type=float, help='weight for ctc loss')
    parser.add_argument('--distance', default='euclidean',
                        help='distance metric for clustering', choices=['cosine', 'euclidean'])
    parser.add_argument('--ctc_distance', default='cosine',
                        help='distance metric for clustering', choices=['cosine', 'euclidean'])

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training & testing')

    parser.add_argument('--quiet', action='store_true',
                        help='quiet mode')

    parser.add_argument('--num_stages', default=4,
                        type=int, help='number of stages')
    parser.add_argument('--num_layers', default=10,
                        type=int, help='number of layers')
    parser.add_argument('--num_f_maps', default=64,
                        type=int, help='number of feature maps')
    parser.add_argument('--features_dim', default=2048,
                        type=int, help='dimension of input features')
    parser.add_argument("--features_path", default=None,
                        type=str, help="path to features")
    parser.add_argument("--gt_path", default="data/{}/groundTruth/",
                        type=str,
                        help="path to ground truth labels")
    parser.add_argument("--base_data_folder",
                        default="./data/",
                        type=str,
                        help="path to data folder")

    parser.add_argument('--bz', default=8, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0005,
                        type=float, help='learning rate')
    parser.add_argument("--num_initial_epochs", default=30, type=int,
                        help="number of epochs for initial training on dense labels")
    parser.add_argument('--num_epochs', default=50,
                        type=int, help='total number of epochs')
    parser.add_argument("--weight_decay", default=0,
                        type=float, help="weight decay")

    parser.add_argument('--ctc_varient', default='ctc',
                        help='ctc variation type', choices=['ctc', 'asg', 'stc'])
    parser.add_argument("--ctc_start_epoch", default=None, type=int,
                        help="epoch to start ctc loss; default: num_initial_epochs")
    parser.add_argument('--ctc_downsample_rate', default=0,
                        type=int, help='downsample rate for ctc')
    parser.add_argument('--stc_insertion_penalty', default=1,
                        type=float, help='insertion penalty for stc')

    parser.add_argument('--project_name', default='outputs',
                        help='wandb project name')
    parser.add_argument('--group', type=str, default=None,
                        help='wandb group name')

    parser.add_argument("--initial_labels_path", default=None,
                        type=str, help="path to initial pseudo labels")
    parser.add_argument("--seed", default=12345,
                        type=int,
                        help="random seed")
    parser.add_argument("--partial_pseudo_labels",
                        action="store_true", help="use partial pseudo labels")
    parser.add_argument("--partial_pseudo_labels_method", default="uncertainty",
                        type=str, help="method for partial pseudo labels", choices=["uncertainty"])
    parser.add_argument("--entropy_weight", default=0,
                        type=float, help="weight for entropy loss")
    parser.add_argument("--smoothness_loss_weight", default=0.15,
                        type=float, help="weight for smoothness loss")
    parser.add_argument("--update_every", default=1, type=int,
                        help="update pseudo labels every n epochs")
    parser.add_argument("--mc_forward_passes", default=10,
                        type=int, help="number of forward passes for MC dropout")
    parser.add_argument("--uncertainty_method", default="entropy", type=str,
                        help="uncertainty estimation method", choices=["entropy", "std", "entropy-2"])
    parser.add_argument("--uncertainty_thresh", default=0.1,
                        type=float, help="uncertainty threshold for pseudo labels")
    parser.add_argument("--uncertainty_temp", default=1,
                        type=float, help="temperature for uncertainty estimation")

    parser.add_argument("--config_file", default=None,
                        type=str, help="path to config file")
    parser.add_argument("--timestamp_path", default=None,
                        type=str, help="path to timestamps")
    parser.add_argument("--confidence_weight", default=0.075,
                        type=float, help="weight for entropy loss")

    parser.add_argument(
        "--loss_weights", help="string of weights for losses", default=None, type=str)
    parser.add_argument("--max_ipw_weight", default=np.inf,
                        type=float, help="max weight for ipw loss")
    parser.add_argument("--loss_normalization", default=False,
                        action="store_true", help="normalize loss by number of frames")
    parser.add_argument("--ce_loss_weight", default=1,
                        type=float, help="weight for cross entropy loss")

    parser.add_argument("--focal_loss", action="store_true",
                        help="use focal loss")
    parser.add_argument("--focal_gamma", default=2,
                        type=float, help="gamma for focal loss")

    parser.add_argument("--wandb_api_key_path", default="wandb_api_key.txt",
                        type=str, help="path to wandb api key file")
    parser.add_argument("--disable_wandb_code",
                        action="store_true", help="disable wandb code upload")
    parser.add_argument("--additional_relaxed_eval",
                        action="store_true", help="use additional relaxed evaluation")
    parser.add_argument("--disable_mixed_precision",
                        action="store_true", help="disable mixed precision training")
    parser.add_argument("--casual_model", action="store_true",
                        help="use casual model")
    parser.add_argument("--save_certain", action="store_true",
                        help="save certain predictions")
    parser.add_argument("--model_dir", default=None,
                        type=str, help="path to model directory")
    parser.add_argument("--save_certain_temp", default=0.5, type=float,
                        help="temperature for saving certain predictions")
    args = parser.parse_args()

    def get_explicit_args(args):
        args_keys = vars(args).keys()
        sys_args = sys.argv[1:]
        explicit_args = {}
        for arg in sys_args:
            if "=" in arg:
                key = arg.split("=")[0]
                if key in args_keys:
                    explicit_args[key] = value
            elif arg.startswith("--"):
                key = arg.replace("--", "")
                if key in args_keys:
                    explicit_args[key] = getattr(args, key)
        return explicit_args

    non_default_args = get_explicit_args(args)

    if args.config_file is not None:
        args = load_config_args(args)

    # update args with non-default args
    for key, value in non_default_args.items():
        setattr(args, key, value)

    if args.loss_weights is not None:
        parsed = False
        try:
            args.loss_weights = ast.literal_eval(args.loss_weights)
            parsed = isinstance(args.loss_weights, dict)  # is dict
            parsed = parsed or (args.loss_weights is None)  # is None
        except:
            pass
        if not parsed and "," in args.loss_weights:
            args.loss_weights = args.loss_weights.split(",")

        logger.info("Loaded loss weights: {}", args.loss_weights)

    if args.casual_model and "casual" not in args.exp_name:
        args.exp_name += " - casual"

    if args.ctc_start_epoch is None:
        args.ctc_start_epoch = args.num_initial_epochs

    return args


def load_config_args(args):

    if args.config_file is None:
        return args

    print(args.config_file)
    assert os.path.exists(args.config_file), "Config file does not exist"

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    folder = os.path.dirname(args.config_file)

    # recursively load config files
    if "config_file" in config:
        args.config_file = os.path.join(folder, config["config_file"])
        args = load_config_args(args)

    for key, value in config.items():
        setattr(args, key, value)

    return args


def run_split(args):

    if not args.quiet:
        if args.wandb_api_key_path is not None:
            with open(args.wandb_api_key_path, "r") as f:
                wandb_api_key = f.read().strip()
            os.environ["WANDB_API_KEY"] = wandb_api_key

        group_name = args.group if args.group is not None else f"{
            args.dataset}_{args.exp_name}"

        wandb.init(project=args.project_name,
                   name=args.exp_name,
                   group=group_name,
                   config=args, reinit=True,
                   settings=wandb.Settings(start_method="fork"),
                   save_code=args.disable_wandb_code,)

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device.index)

    num_stages = args.num_stages
    num_layers = args.num_layers
    num_f_maps = args.num_f_maps
    features_dim = args.features_dim
    bz = args.bz
    lr = args.lr
    num_epochs = args.num_epochs

    sample_rate = 1

    if args.dataset == "cholec80":
        args.additional_relaxed_eval = True

    vid_list_file = args.base_data_folder+args.dataset + \
        "/splits/train.split"+args.split+".bundle"
    vid_list_file_val = args.base_data_folder + \
        args.dataset+"/splits/val.split"+args.split+".bundle"
    vid_list_file_tst = args.base_data_folder+args.dataset + \
        "/splits/test.split"+args.split+".bundle"
    if args.features_path is None:
        features_path = args.base_data_folder+args.dataset+"/features/"
    elif "{}" in args.features_path:
        features_path = args.features_path.format(args.dataset)
    else:
        features_path = args.features_path

    if args.gt_path is None:
        gt_path = args.base_data_folder+args.dataset+"/groundTruth/"
    elif "{}" in args.gt_path:
        gt_path = args.gt_path.format(args.dataset)
    else:
        gt_path = args.gt_path

    if args.initial_labels_path is not None and "{}" in args.initial_labels_path:
        args.initial_labels_path = args.initial_labels_path.format(args.split)

    mapping_file = args.base_data_folder+args.dataset+"/mapping.txt"

    bz_stages = args.exp_name
    if args.model_dir is not None:
        model_dir = args.model_dir
    else:
        model_dir = "./models/" + args.dataset + \
            f"/{bz_stages}/split_" + args.split
    results_dir = "./results/" + args.dataset + \
        f"/{bz_stages}/split_" + args.split

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("{} dataset {} in split {} for single stamp supervision".format(
        args.action, args.dataset, args.split))
    print('batch size is {}, number of stages is {}, sample rate is {}\n'.format(
        bz, num_stages, sample_rate))

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    # check if args.loss_weights is a dict
    if args.loss_weights is not None:
        if isinstance(args.loss_weights, dict):
            # convert to list
            args.loss_weights = [float(args.loss_weights[key])
                                 for key in actions_dict.keys()]
        elif isinstance(args.loss_weights, list):
            args.loss_weights = [float(w) for w in args.loss_weights]

    num_classes = len(actions_dict)
    writer = SummaryWriter(f"runs/{args.dataset}_{args.exp_name}_{args.split}")
    trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes,
                      ctc_weight=args.ctc_weight,
                      entropy_weight=args.entropy_weight,
                      smoothness_loss_weight=args.smoothness_loss_weight,
                      ctc_downsample_rate=args.ctc_downsample_rate,
                      ctc_varient=args.ctc_varient,
                      stc_insertion_penalty=args.stc_insertion_penalty,
                      ctc_distance=args.ctc_distance,
                      use_initial_psuedo_labels=args.initial_labels_path is not None,
                      partial_pseudo_labels=args.partial_pseudo_labels,
                      partial_pseudo_labels_method=args.partial_pseudo_labels_method,
                      update_every=args.update_every,
                      mc_forward_passes=args.mc_forward_passes,
                      uncertainty_method=args.uncertainty_method,
                      uncertainty_thresh=args.uncertainty_thresh,
                      confidence_weight=args.confidence_weight,
                      loss_weights=args.loss_weights,
                      focal_loss=args.focal_loss,
                      focal_gamma=args.focal_gamma,
                      max_ipw_weight=args.max_ipw_weight,
                      disable_mixed_precision=args.disable_mixed_precision,
                      is_casual=args.casual_model,
                      loss_normalization=args.loss_normalization,
                      temp=args.uncertainty_temp,
                      ce_loss_weight=args.ce_loss_weight,)

    if args.action == "train":
        batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, args.dataset, features_path, sample_rate,
                                   distance=args.distance, initial_labels_path=args.initial_labels_path,
                                   partial_pseudo_labels=args.partial_pseudo_labels,
                                   timestamp_path=args.timestamp_path,

                                   )
        batch_gen.read_data(vid_list_file)

        # Train the model
        trainer.train(model_dir,
                      batch_gen,
                      writer,
                      initial_epochs=args.num_initial_epochs,
                      num_epochs=num_epochs,
                      batch_size=bz,
                      learning_rate=lr,
                      device=device,
                      upload=not args.quiet,
                      ctc_start_epoch=args.ctc_start_epoch,)

    if args.save_certain:
        trainer.predict(model_dir, results_dir, features_path,
                        vid_list_file, num_epochs, actions_dict,
                        device, sample_rate,
                        dataset=args.dataset,
                        save_certain=args.save_certain,
                        certain_temp=args.save_certain_temp,)

    # Predict the output label for each frame in evaluation and output them
    trainer.predict(model_dir, results_dir, features_path,
                    vid_list_file_tst, num_epochs, actions_dict,
                    device, sample_rate,
                    dataset=args.dataset,
                    save_certain=args.save_certain,
                    certain_temp=args.save_certain_temp,)

    # Read output files and measure metrics (F1@10, 25, 50, Edit, Acc)
    metrics = evaluate(args.dataset, args.split, args.exp_name, args.base_data_folder, gt_path,
                       additional_relaxed_eval=args.additional_relaxed_eval)

    if os.path.exists(vid_list_file_val):
        trainer.predict(model_dir, results_dir, features_path,
                        vid_list_file_val, num_epochs, actions_dict,
                        device, sample_rate,
                        dataset=args.dataset,
                        )

        # Read output files and measure metrics (F1@10, 25, 50, Edit, Acc)
        metrics_val = evaluate(args.dataset, args.split, args.exp_name, args.base_data_folder, gt_path,
                               additional_relaxed_eval=args.additional_relaxed_eval,
                               parition="val")

    if not args.quiet:
        # Log metrics to wandb

        if os.path.exists(vid_list_file_val):
            for key, value in metrics_val.items():
                wandb.log({f"val_{key}": value})

            for key, value in metrics.items():
                wandb.log({f"test_{key}": value})
        else:
            for key, value in metrics.items():
                wandb.log({f"val_{key}": value})

    return metrics


def average_metrics(metrics):
    avg_metrics = {}
    for metric in metrics[0].keys():
        avg_metrics[metric] = np.mean([m[metric] for m in metrics])
    return avg_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(args):

    set_seed(args.seed)

    metrics = []

    splits = [args.split]

    for split in splits:
        args.split = str(split)
        results = run_split(args)
        metrics.append(results)

        print("Split {split}".format(split=split))
        print(results)

    avg_metrics = average_metrics(metrics)
    print(avg_metrics)


if __name__ == '__main__':

    args = get_args()
    main(args)
