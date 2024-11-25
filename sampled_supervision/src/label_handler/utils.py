import numpy as np


def sample_label(start_boundery, end_boundery):
    return np.random.randint(start_boundery, end_boundery)

def center_label(start_boundery, end_boundery):
    return int((start_boundery + end_boundery) // 2)

def get_uniform_regions(global_start, global_end, region_num):
    regions = []
    region_size = int((global_end - global_start) / region_num)
    for i in range(region_num):
        regions.append([global_start + i * region_size, global_start + (i + 1) * region_size])
    
    # fix to assure all the regions are covered
    regions[-1][1] = global_end

    return regions

def get_uniform_regions_with_noise(global_start, global_end, region_num, max_noise):
    # get uniform regions up to noise
    regions = get_uniform_regions(global_start, global_end, region_num)

    # add noise to the regions
    for region in regions:
        region[0] += np.random.randint(-max_noise, max_noise)
        region[1] += np.random.randint(-max_noise, max_noise)

    # fix to assure all the regions are covered
    regions[0][0] = global_start
    regions[-1][1] = global_end

    return regions


def get_uniform_samples(regions):
    samples = []
    for region in regions:
        samples.append(sample_label(region[0], region[1]))

    samples = np.array(samples)
    return samples


def get_fixed_samples(regions):
    samples = []
    for region in regions:
        samples.append(center_label(region[0], region[1]))

    samples = np.array(samples)
    return samples