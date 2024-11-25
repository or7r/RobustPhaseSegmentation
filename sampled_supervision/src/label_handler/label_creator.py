import torch
import numpy as np
import os

from .utils import get_uniform_regions, get_uniform_samples, get_fixed_samples

class LabelCreator:
    def __init__(self, vid_list_file, gt_path, actions_dict, sample_rate, fps=15) -> None:
        self.actions_dict = actions_dict
        self.sample_rate = sample_rate
        self.list_of_examples = self.read_example_list(vid_list_file)
        self.gt = self.read_labels(gt_path)
        self.fps = 15
        

        pass

    def read_labels(self, labels_path):
        gt = {}
        for vid in self.list_of_examples:
            file_ptr = open(labels_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(len(content))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            classes = classes[::self.sample_rate]
            gt[vid] = classes
        
        return gt

    def read_example_list(self, vid_list_file):
        with open(vid_list_file, 'r') as file_ptr:
            list_of_examples = file_ptr.read().split('\n')[:-1]
        return list_of_examples
    
    def get_sampled_labels(self, region_num):

        sampled_labels = {}
        
        for vid in self.list_of_examples:
            gt = self.gt[vid]
            regions = get_uniform_regions(0, len(gt), region_num)
            samples = get_uniform_samples(regions)
            sampled_labels[vid] = samples


        return sampled_labels
    
    def get_fixed_labels(self, region_num):

        sampled_labels = {}
        
        for vid in self.list_of_examples:
            gt = self.gt[vid]
            regions = get_uniform_regions(0, len(gt), region_num)
            samples = get_fixed_samples(regions)
            sampled_labels[vid] = samples


        return sampled_labels

    def get_random_labels(self, region_num):
        sampled_labels = {}

        for vid in self.list_of_examples:
            gt = self.gt[vid]
            sampled_labels[vid] = np.random.choice(len(gt), size=region_num, replace=False)
            # sort
            sampled_labels[vid].sort()
        
        return sampled_labels

    def save_samples(self, sampled_labels, save_path):
        # should be called after get_sampled_labels,
        # people will use this line to read it
        # random_index = np.load(gt_path + dataset_name + "_annotation_all.npy", allow_pickle=True).item()
        # where random_index is dict of numpy arrays
        # 
        save_path_dir = os.path.dirname(save_path) 
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)

        np.save(save_path, sampled_labels, allow_pickle=True)
    
    def add_artificial(self, sampled_labels, artificial_num=0):
        # this function will add num_artificial labels copies of each label, in quater a second intervals
        # this is to simulate the effect of having more labels
        # this will be used to test the effect of having more labels

        interval_length = 0.25 * self.sample_rate

        for vid in sampled_labels:
            labels = sampled_labels[vid]
            new_labels = []
            for label in labels:
                for i in range(artificial_num + 1):
                    new_labels.append(label + (i * interval_length) // 2)
                    new_labels.append(label - (i * interval_length) // 2)
            new_labels = list(set(new_labels))
            # sort  
            new_labels.sort()
            # drop the ones that are out of bounds
            new_labels = [int(label) for label in new_labels if label >= 0 and label < len(self.gt[vid])]
            sampled_labels[vid] = np.array(new_labels)
        
        return sampled_labels