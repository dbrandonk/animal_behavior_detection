import numpy as np

def data_aug_mars():

    train_data = np.load('data/aicrowd1/train.npy',allow_pickle=True).item()

    SEQ_LEN = 90

    data = None
    target = None

    for k, seq in train_data['sequences'].items():

        kp = seq['keypoints']
        annot = seq['annotations']

        padding = np.zeros(((SEQ_LEN - 1), kp.shape[1], kp.shape[2], kp.shape[3]))
        seq_list = np.vstack((padding, kp))

        seq_list = np.lib.stride_tricks.sliding_window_view(seq_list, SEQ_LEN, axis=0)
        # seq_list = np.moveaxis(seq_list, [4,0,1,2,3], [0,1,2,3,4])
        seq_list = np.moveaxis(seq_list, [4,1,2,3], [1,2,3,4])
        seq_list = seq_list.reshape(seq_list.shape[0], seq_list.shape[1], (seq_list.shape[2] * seq_list.shape[3] * seq_list.shape[4]))

        if data is None:
            data = seq_list.copy()
            target = annot.copy()

        else:
            data = np.concatenate((data, seq_list), axis = 0)
            target = np.concatenate((target, annot), axis = 0)

    np.save('data/aicrowd1/train_seq_data.npy', data)
    np.save('data/aicrowd1/train_seq_target.npy', target)

if __name__ == '__main__':
    data_aug_mars()
