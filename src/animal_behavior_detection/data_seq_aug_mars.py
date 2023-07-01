"""
Data Augmentation for Mars Dataset
"""
import numpy as np

SEQ_LEN = 90


def data_aug_mars():
    """
    Perform data augmentation for the Mars dataset.

    This function loads the training data from the 'train.npy' file and performs data augmentation
    by sliding a window over the sequences and reshaping them. The augmented data is saved as
    'train_seq_data.npy', and the corresponding annotations are saved as 'train_seq_target.npy'.
    """

    train_data = np.load('data/train.npy', allow_pickle=True).item()

    data = None
    target = None

    for _, seq in train_data['sequences'].items():

        key_point = seq['keypoints']
        annot = seq['annotations']

        padding = np.zeros(
            ((SEQ_LEN - 1),
             key_point.shape[1],
             key_point.shape[2],
             key_point.shape[3]))
        seq_list = np.vstack((padding, key_point))

        seq_list = np.lib.stride_tricks.sliding_window_view(
            seq_list, SEQ_LEN, axis=0)
        seq_list = np.moveaxis(seq_list, [4, 1, 2, 3], [1, 2, 3, 4])
        seq_list = seq_list.reshape(
            seq_list.shape[0],
            seq_list.shape[1],
            (seq_list.shape[2] *
             seq_list.shape[3] *
             seq_list.shape[4]))

        if data is None:
            data = seq_list.copy()
            target = annot.copy()

        else:
            data = np.concatenate((data, seq_list), axis=0)
            target = np.concatenate((target, annot), axis=0)

    np.save('data/train_seq_data.npy', data)
    np.save('data/train_seq_target.npy', target)


if __name__ == '__main__':
    data_aug_mars()
