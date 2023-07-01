"""
Behavior Analysis Module
This module contains functions for behavior analysis and visualization of annotation data.
"""

from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def data_sample_count_mars():
    """
    Calculate Class Sample Count

    This function calculates the sample count for each class in the training data.

    Returns:
        numpy.ndarray: An array containing the sample count for each class.
    """

    train_data = np.load('data/train.npy', allow_pickle=True).item()

    data = None
    target = None

    for _, seq in train_data['sequences'].items():
        key_point = seq['keypoints']
        annot = seq['annotations']

        if data is None:
            data = key_point.copy()
            target = annot.copy()
        else:
            data = np.concatenate((data, key_point), axis=0)
            target = np.concatenate((target, annot), axis=0)

    class_sample_count = []

    for class_type in np.unique(target):
        class_sample_count.append(len(np.where(target == class_type)[0]))

    class_sample_count = np.array(class_sample_count)

    return class_sample_count


def num_to_text(anno_list):
    """
    Convert Numeric Annotations to Text

    This function converts a list of numeric annotations into their corresponding text
    representations based on the vocabulary provided in the training data.

    Args:
        anno_list (list): The list of numeric annotations to be converted.

    Returns:
        numpy.ndarray: An array containing the text representations of the numeric annotations.
    """

    train = np.load('data/train.npy', allow_pickle=True).item()
    number_to_class = dict(enumerate(train['vocabulary']))
    return np.vectorize(number_to_class.get)(anno_list)


def plot_annotation_strip(  # pylint: disable=too-many-locals
        annotation_sequence, start_frame=0, stop_frame=100,
        title="Behavior Labels"):
    """
    Plot Annotation Strip

    This function plots the annotation strip for a given annotation sequence.
    It visualizes the annotations as a strip of colored blocks, where each
    block represents a specific behavior class. The strip starts from the
    specified start_frame and ends at the specified stop_frame.

    Args:
        annotation_sequence (list): The annotation sequence to be plotted.
        start_frame (int, optional): The starting frame index for the strip. Defaults to 0.
        stop_frame (int, optional): The ending frame index for the strip. Defaults to 100.
        title (str, optional): The title of the plot. Defaults to "Behavior Labels".

    """

    train = np.load('data/train.npy', allow_pickle=True).item()
    class_to_color = {
        'other': 'white',
        'attack': 'red',
        'mount': 'green',
        'investigation': 'orange'}
    class_to_number = {s: i for i, s in enumerate(train['vocabulary'])}

    annotation_num = []
    for item in annotation_sequence[start_frame:stop_frame]:
        annotation_num.append(class_to_number[item])

    all_classes = list(set(annotation_sequence[start_frame:stop_frame]))

    cmap = colors.ListedColormap(['red', 'orange', 'green', 'white'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    height = 200
    arr_to_plot = np.repeat(
        np.array(annotation_num)[
            :,
            np.newaxis].transpose(),
        height,
        axis=0)

    _, axis = plt.subplots(figsize=(16, 3))
    axis.imshow(arr_to_plot, interpolation='none', cmap=cmap, norm=norm)

    axis.set_yticks([])
    axis.set_xlabel('Frame Number')
    plt.title(title)

    legend_patches = []
    for item in all_classes:
        legend_patches.append(
            mpatches.Patch(
                color=class_to_color[item],
                label=item))

    plt.legend(
        handles=legend_patches,
        loc='center left',
        bbox_to_anchor=(
            1,
            0.5))

    plt.tight_layout()

    plt.savefig('./plots/class_imbalance_strip.png')
    plt.close()
    plt.cla()
    plt.clf()


def plot_video():
    """
    Plot Video Annotation Strip
    """

    train = np.load('data/train.npy', allow_pickle=True).item()

    for _, seq in train['sequences'].items():
        annot = seq['annotations']

        text_sequence = num_to_text(annot)

        plot_annotation_strip(
            text_sequence,
            start_frame=0,
            stop_frame=len(annot) + 1000)
        break


def main():
    """main"""

    classes = ['attack', 'investigation', 'mount', 'other']
    class_sample_count = data_sample_count_mars()

    plt.bar(classes, class_sample_count)
    plt.xlabel('CLASSES')
    plt.ylabel('No. of INSTANCES')
    plt.title('CLASS IMBALANCE')
    plt.savefig('./plots/class_imbalance_bar.png')
    plt.close()
    plt.cla()
    plt.clf()

    plot_video()


if __name__ == '__main__':
    main()
