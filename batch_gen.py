#!/usr/bin/python3.6

from copy import deepcopy
import torch
import numpy as np
import random
import torch.nn.functional as F
from utils import get_distance_func


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, dataset, features_path, sample_rate, shuffle=True,
                 distance='euclidean', initial_labels_path=None, timestamp_path=None,
                 partial_pseudo_labels=False,):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.gt = {}
        self.confidence_mask = {}
        self.shuffle = shuffle
        self.distance = get_distance_func(distance)

        dataset_name = dataset  # gt_path.split('/')[2]
        if timestamp_path is None:
            timestamp_path = gt_path + dataset_name + "_annotation_all.npy"
        self.random_index = np.load(timestamp_path, allow_pickle=True).item()

        if partial_pseudo_labels:
            self.pseudo_labels = dict()

        self.initial_labels_path = initial_labels_path

    def reset(self):
        self.index = 0
        if self.shuffle:
            random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]

        if set(self.list_of_examples) != set(self.random_index.keys()):
            print("Error: Random index does not match with the list of examples")
            print("Droping the examples that are not in the random index")
            self.list_of_examples = list(
                set(self.list_of_examples).intersection(set(self.random_index.keys())))

        file_ptr.close()
        if self.shuffle:
            random.shuffle(self.list_of_examples)
        self.generate_confidence_mask()

        if self.initial_labels_path is not None:
            self.initial_gt = self.read_labels(
                self.initial_labels_path, additional_tokens={"uncertain": -100})

    def read_labels(self, labels_path, additional_tokens=None):
        actions_dict = self.actions_dict
        if additional_tokens is not None:
            actions_dict = deepcopy(self.actions_dict)
            actions_dict.update(additional_tokens)
        gt = {}
        for vid in self.list_of_examples:
            file_ptr = open(labels_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(len(content))
            for i in range(len(classes)):
                classes[i] = actions_dict[content[i]]
            classes = classes[::self.sample_rate]
            gt[vid] = classes

        return gt

    def generate_confidence_mask(self):
        for vid in self.list_of_examples:
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(len(content))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            classes = classes[::self.sample_rate]
            self.gt[vid] = classes.astype(int)
            num_frames = classes.shape[0]

            random_idx = self.random_index[vid]

            if hasattr(self, 'pseudo_labels'):
                self.pseudo_labels[vid] = torch.ones(
                    classes.shape[0], dtype=torch.long) * (-100)
                for t in random_idx:
                    self.pseudo_labels[vid][t] = int(classes[t])

            # Generate mask for confidence loss. There are two masks for both side of timestamps
            left_mask = np.zeros([self.num_classes, num_frames - 1])
            right_mask = np.zeros([self.num_classes, num_frames - 1])
            for j in range(len(random_idx) - 1):
                left_mask[int(classes[random_idx[j]]),
                          random_idx[j]:random_idx[j + 1]] = 1
                right_mask[int(classes[random_idx[j + 1]]),
                           random_idx[j]:random_idx[j + 1]] = 1

            self.confidence_mask[vid] = np.array([left_mask, right_mask])

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_confidence = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(self.gt[vid])
            batch_confidence.append(self.confidence_mask[vid])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[
                                         0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(
            length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(
            length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[
                1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[
                0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(
                self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask, batch_confidence

    def get_single_random(self, batch_size, max_frames):
        # Generate target for only timestamps. Do not generate pseudo labels at first 30 epochs.
        batch = self.list_of_examples[self.index - batch_size:self.index]
        boundary_target_tensor = torch.ones(
            len(batch), max_frames, dtype=torch.long) * (-100)
        for b, vid in enumerate(batch):
            single_frame = self.random_index[vid]
            gt = self.gt[vid]
            frame_idx_tensor = torch.from_numpy(np.array(single_frame))
            gt_tensor = torch.from_numpy(gt.astype(int))
            boundary_target_tensor[b,
                                   frame_idx_tensor] = gt_tensor[frame_idx_tensor]

        return boundary_target_tensor

    def get_initial_labels(self, batch_size, max_frames):
        assert self.initial_labels_path is not None
        batch = self.list_of_examples[self.index - batch_size:self.index]
        boundary_target_tensor = torch.ones(
            len(batch), max_frames, dtype=torch.long) * (-100)
        for b, vid in enumerate(batch):
            initial_gt = self.initial_gt[vid]
            initial_gt_tensor = torch.from_numpy(initial_gt.astype(int))
            boundary_target_tensor[b, :initial_gt.shape[0]] = initial_gt_tensor

        return boundary_target_tensor

    def get_boundary(self, batch_size, pred):
        # This function is to generate pseudo labels

        batch = self.list_of_examples[self.index - batch_size:self.index]
        num_video, _, max_frames = pred.size()
        boundary_target_tensor = torch.ones(
            num_video, max_frames, dtype=torch.long) * (-100)

        distance = self.distance

        for b, vid in enumerate(batch):
            single_idx = self.random_index[vid]
            vid_gt = self.gt[vid]
            features = pred[b]
            boundary_target = np.ones(vid_gt.shape) * (-100)
            # frames before first single frame has same label
            boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]
            left_bound = [0]

            # Forward to find action boundaries
            for i in range(len(single_idx) - 1):
                start = single_idx[i]
                end = single_idx[i + 1] + 1
                left_score = torch.zeros(end - start - 1, dtype=torch.float)
                for t in range(start + 1, end):
                    center_left = torch.mean(features[:, start:t], dim=1)
                    score_left = distance(features[:, start:t].transpose(1, 0),
                                          center_left.reshape(-1, 1).transpose(1, 0)).mean()
                    # diff_left = features[:, start:t] - center_left.reshape(-1, 1)
                    # score_left = torch.mean(torch.norm(diff_left, dim=0))

                    center_right = torch.mean(features[:, t:end], dim=1)
                    score_right = distance(features[:, t:end].transpose(1, 0),
                                           center_right.reshape(-1, 1).transpose(1, 0)).mean()
                    # diff_right = features[:, t:end] - center_right.reshape(-1, 1)
                    # score_right = torch.mean(torch.norm(diff_right, dim=0))

                    left_score[t-start-1] = ((t-start) * score_left +
                                             (end - t) * score_right)/(end - start)

                cur_bound = torch.argmin(left_score) + start + 1
                left_bound.append(cur_bound.item())

            # Backward to find action boundaries
            right_bound = [vid_gt.shape[0]]
            for i in range(len(single_idx) - 1, 0, -1):
                start = single_idx[i - 1]
                end = single_idx[i] + 1
                right_score = torch.zeros(end - start - 1, dtype=torch.float)
                for t in range(end - 1, start, -1):
                    center_left = torch.mean(features[:, start:t], dim=1)
                    score_left = distance(features[:, start:t].transpose(1, 0),
                                          center_left.reshape(-1, 1).transpose(1, 0)).mean()
                    # diff_left = features[:, start:t] - center_left.reshape(-1, 1)
                    # score_left = torch.mean(torch.norm(diff_left, dim=0))

                    center_right = torch.mean(features[:, t:end], dim=1)
                    score_right = distance(features[:, t:end].transpose(1, 0),
                                           center_right.reshape(-1, 1).transpose(1, 0)).mean()
                    # diff_right = features[:, t:end] - center_right.reshape(-1, 1)
                    # score_right = torch.mean(torch.norm(diff_right, dim=0))

                    right_score[t-start-1] = ((t-start) * score_left +
                                              (end - t) * score_right)/(end - start)

                cur_bound = torch.argmin(right_score) + start + 1
                right_bound.append(cur_bound.item())

            # Average two action boundaries for same segment and generate pseudo labels
            left_bound = left_bound[1:]
            right_bound = right_bound[1:]
            num_bound = len(left_bound)
            for i in range(num_bound):
                temp_left = left_bound[i]
                temp_right = right_bound[num_bound - i - 1]
                middle_bound = int((temp_left + temp_right)/2)
                boundary_target[single_idx[i]                                :middle_bound] = vid_gt[single_idx[i]]
                boundary_target[middle_bound:single_idx[i +
                                                        1] + 1] = vid_gt[single_idx[i + 1]]

            # frames after last single frame has same label
            boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]
            boundary_target_tensor[b, :vid_gt.shape[0]
                                   ] = torch.from_numpy(boundary_target)

        return boundary_target_tensor

    def get_boundary_with_uncertainty(self, batch_size, max_frames):
        # Return pseudo labels of the last batch
        batch = self.list_of_examples[self.index - batch_size:self.index]

        # batch_pseudo_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        batch_pseudo_tensor = torch.ones(
            len(batch), max_frames, dtype=torch.long) * (-100)
        for b, vid in enumerate(batch):
            batch_pseudo_tensor[b, :self.pseudo_labels[vid].shape[0]
                                ] = self.pseudo_labels[vid]

        return batch_pseudo_tensor

    # This function was adpated from https://github.com/xmed-lab/TimeStamp-Surgical/blob/main/batch_gen.py
    def update_pseudo_labels(self, batch_size, pseudo_labels, uncertainty_scores, thres,):
        # self.max_extend += 0.2
        # batch = self.list_of_samples[self.last_index: self.index]
        batch = self.list_of_examples[self.index - batch_size:self.index]

        for b, vid in enumerate(batch):
            timestamp = self.random_index[vid]
            video_gt = self.gt[vid]
            seq_length = video_gt.shape[0]
            pseudo_label = pseudo_labels[b].squeeze(0)
            uncertainty_score = uncertainty_scores[b].squeeze(0)
            self.pseudo_labels[vid] = torch.ones(
                seq_length, dtype=torch.long) * (-100)
            for i in range(len(timestamp)):
                left_bound = timestamp[i] - 1
                right_bound = timestamp[i] + 1
                L = 0 if i == 0 else timestamp[i-1]+1
                R = seq_length-1 if i == len(timestamp)-1 else timestamp[i+1]-1
                while left_bound >= L and ((pseudo_label[left_bound] == video_gt[timestamp[i]] and uncertainty_score[left_bound] < thres)):  # \
                    # or (pseudo_label[left_bound] != video_gt[timestamp[i]] and uncertainty_score[left_bound] > 1)): # \
                    # and torch.sum(pseudo_label[max(L, left_bound-15):left_bound]==video_gt[timestamp[i]])/(left_bound-max(L, left_bound-15))>0.6)):
                    left_bound -= 1
                while right_bound <= R and ((pseudo_label[right_bound] == video_gt[timestamp[i]] and uncertainty_score[right_bound] < thres)):  # \
                    # or (pseudo_label[right_bound] != video_gt[timestamp[i]] and uncertainty_score[right_bound] > 1)): \
                    # and torch.sum(pseudo_label[right_bound:min(R, right_bound+15)]==video_gt[timestamp[i]])/(min(R, right_bound+15)-right_bound)>0.6)):
                    right_bound += 1
                # left_bound = max(left_bound, timestamp[i] - int(self.max_extend * (timestamp[i]-L)))
                # right_bound = min(right_bound, timestamp[i] + int(self.max_extend * (R-timestamp[i])))
                self.pseudo_labels[vid][left_bound +
                                        1: right_bound] = int(video_gt[timestamp[i]])

    def create_output_placeholder(self, batch_size, logits):
        batch = self.list_of_examples[self.index - batch_size: self.index]
        num_video, _, max_frames = logits.size()
        target_tensor = torch.ones(
            num_video, max_frames, dtype=torch.long) * (-100)
        return batch, target_tensor

    def calc_ipw_weights_min_normalization(self):
        # Use random_index to calcualte loss weights
        # First, we will calculate the number of occurence of each class
        num_occurence = torch.zeros(self.num_classes)
        for vid in self.list_of_examples:
            gt = self.gt[vid]
            for i in range(len(gt)):
                num_occurence[int(gt[i])] += 1

        ipw_weights = 1 / (num_occurence + 1e-8)

        # Rescale weights to be >= 1
        m = torch.min(ipw_weights)
        ipw_weights = ipw_weights / m

        return ipw_weights

    def calc_ipw_weights(self, apply_root=False, apply_log=False):
        # Use random_index to calcualte loss weights
        # First, we will calculate the number of occurence of each class
        num_occurence = torch.zeros(self.num_classes)
        for vid in self.list_of_examples:
            gt = self.gt[vid]
            timestamp = self.random_index[vid]
            gt = gt[timestamp]
            for i in range(len(gt)):
                num_occurence[int(gt[i])] += 1

        prob = num_occurence / torch.sum(num_occurence)

        ipw_weights = 1 / (prob + 1e-8)

        if apply_root:
            ipw_weights = torch.sqrt(ipw_weights)

        if apply_log:
            ipw_weights = torch.log(ipw_weights)
        # # Rescale weights to be >= 1
        # m  = torch.min(ipw_weights)
        # ipw_weights = ipw_weights / m

        return ipw_weights
