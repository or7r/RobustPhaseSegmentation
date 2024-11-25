import matplotlib.pyplot as plt
import numpy as np
import os


def segment_bars(save_path, *labels):
    num_pics = len(labels)
    color_map = plt.get_cmap('seismic')
    # color_map =
    fig = plt.figure(figsize=(15, num_pics * 1.5))

    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=20)

    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1,  i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow([label], **barprops)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def plot_video_gestures(plot_name, gestures_list, names, output_dir):
    """
    :param plot_name: 
    :param gestures_list:
    :param names:
    :return:
    """
    fig, axs = plt.subplots(len(gestures_list), 1,
                            figsize=(7,  0.7*len(gestures_list)))
    # , figsize = figsize,
    # gridspec_kw = {'wspace': 0, 'hspace': 0}

    fig.suptitle(f'{plot_name}', fontsize=10)

    for i, gestures in enumerate(gestures_list):
        map = np.tile(gestures, (100, 1))
        axs[i].axis('off')
        axs[i].set_title(names[i], fontsize=10, pad=10)
        axs[i].imshow(map, cmap='tab20', aspect="auto")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, plot_name+".png"))
    plt.show()
