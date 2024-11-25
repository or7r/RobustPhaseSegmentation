import h5py
import numpy as np
import os
import argparse


def reformat_features(data_file, output_dir, vid_format="VID{:02}.npy"):

    f = h5py.File(data_file, 'r')
    frame_id = f['frame_id'][:]
    video_id = f['video_id'][:]
    embeddings = f['embeddings'][:]

    video_ids = np.unique(video_id)
    for idx in video_ids:
        video_mask = video_id == idx
        video_frame_id = frame_id[video_mask]
        video_embeddings = embeddings[video_mask]
        # Sort by frame_id
        _, order = np.unique(video_frame_id, return_index=True)
        # order = np.argsort(video_frame_id)
        video_frame_id = video_frame_id[order]
        video_embeddings = video_embeddings[order]
        
        # Save to file
        output_file = os.path.join(output_dir, vid_format.format(idx))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, video_embeddings.T)


def reformat_all_features(data_dir, output_dir, vid_format):
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".hdf5"):
                data_file = os.path.join(root, file)
                reformat_features(data_file, output_dir, vid_format) 

class MultiBypass140Formater:
    @staticmethod
    def format(idx):
        if idx < 100:
            return 'BBP{:02}.npy'.format(idx)
        else:
            return 'SBP{:02}.npy'.format(idx - 100)


def get_format(dataset):
    if dataset == "cholec80":
        return "video-{}.npy"
    elif dataset == "MultiBypass140":
        return MultiBypass140Formater()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Reformat features')
    parser.add_argument('--feature_extract_path', type=str, help='path to feature extraction folder', required=True)
    parser.add_argument('--output_dir', type=str, help='path to output', default=None)
    parser.add_argument('--dataset', type=str, help='dataset name', choices=['cholec80', 
                                                                             "MultiBypass140"])
    parser.add_argument('--base_dataset', type=str, help='base dataset name', default="cholec80",
                        choices=['cholec', 'imagenet'])  # Just for naming
    parser.add_argument('--run_id', type=int, help='run id', default=4)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"features/{args.dataset}/{args.dataset}-{args.base_dataset}{args.run_id:03}"
        

    os.makedirs(args.output_dir, exist_ok=True)


    data_dir = os.path.join(args.feature_extract_path,
                             f'runs/{args.dataset}/finetuning/{args.base_dataset}_to_cholec/series_01/run_{args.run_id:03}/test/phase/100/0/extracted_features_Head/')

    vid_format = get_format(args.dataset)
    reformat_all_features(data_dir, args.output_dir, vid_format)