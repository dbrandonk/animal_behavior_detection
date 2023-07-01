"""
This module provides utility functions and classes for training and
evaluating a Transformer-based neural network model.

Classes:
    - MeanMeter: Utility class for computing and tracking the mean value.
    - PositionalEncoding: Module for injecting positional encodings into input sequences.
    - Transformer: Transformer model implementation.

Functions:
    - data_aug_mars: Perform data augmentation for the Mars dataset.
    - data_load_mars_seq: Load Mars sequential data and create data loaders.
    - plot_perf:
        Plot performance metrics (loss, accuracy, F1 score, precision).
    - random_perf:
        Calculate performance metrics (accuracy, F1 score, precision) of random prediction baseline.
    - run: Execute the complete training and evaluation pipeline.

Constants:
    - SEQ_LEN: The length of the input sequences.


Note:
    This module assumes the existence of certain data files in specific paths
    ('data/aicrowd1/train.npy' and 'data/train_seq_data.npy').
    Make sure to have these files available or modify the code accordingly.
"""

import copy
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import TensorDataset, DataLoader

SEQ_LEN = 90


class MeanMeter():
    """
    Utility class for computing and tracking the mean value.
    This class provides functionality to compute and track the mean value of a series of values.
    """

    def __init__(self):
        """
        Initializes a new instance of the MeanMeter class.
        """
        self.val = 0
        self.mean = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """
        Resets the meter.
        """
        self.val = 0
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, val, input_count=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The new value.
            input_count (int): The count of the new value. Default is 1.
        """
        self.val = val
        self.sum += val * input_count
        self.count += input_count
        self.mean = self.sum / self.count


class PositionalEncoding(torch.nn.Module):

    """
    PositionalEncoding module injects some information about the
    relative or absolute position of the tokens in the sequence. The
    positional encodings have the same dimension as the features so that
    the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
    different frequencies.
    """

    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 1000):
        """
        Initialize the PositionalEncoding module and its parameters.

        Args:
            d_model (int): The dimensionality of the input features and positional encodings.
            dropout (float, optional): The dropout rate to apply during training. Defaults to 0.1.
            max_len (int, optional): The maximum length of the input sequences. Defaults to 1000.
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # pylint: disable=no-member
        div_term = torch.exp(torch.arange(0, d_model, 2) *  # pylint: disable=no-member
                             (-math.log(10000.0) / d_model))
        positional_encoding = torch.zeros(max_len, 1, d_model)  # pylint: disable=no-member
        positional_encoding[:, 0, 0::2] = torch.sin(  # pylint: disable=no-member
            position * div_term)
        positional_encoding[:, 0, 1::2] = torch.cos(  # pylint: disable=no-member
            position * div_term)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, input_):
        """
        Perform a forward pass of the PositionalEncoding module.

        Args:
            input_ (torch.Tensor): The input tensor of shape [seq_len, batch_size, feature].

        Returns:
            torch.Tensor: The output tensor after adding positional encodings and applying dropout.

        """
        input_ = input_ + self.pe[:input_.size(0)]
        return self.dropout(input_)


class Transformer(torch.nn.Module):
    """
    Transformer Model

    This class defines a transformer-based neural network model.

    Methods:
        init_weights(): Initialize the weights of the fully connected layers.
        forward(input_): Perform a forward pass of the Transformer model.

    """

    def __init__(self):
        """Initialize the Transformer model."""
        super().__init__()

        d_model = 28
        nhead = 4
        nlayers = 4

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.fc1 = torch.nn.Linear(2520, 512)
        self.fc2 = torch.nn.Linear(512, 4)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize Model Weights."""
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_):
        """
        Forward Pass

        forward pass of the model.
        It takes an input tensor and applies the positional encoding layer,
        the transformer encoder, and the fully connected layers to produce the output.

        Args:
            input_ (torch.Tensor): The input tensor to the model.

        Returns:
            torch.Tensor: The output tensor produced by the model.

        """

        out = self.pos_encoder(input_)
        out = self.transformer_encoder(out)
        out = out.permute((1, 0, 2))
        out = torch.flatten(out, 1)  # pylint: disable=no-member
        out = torch.nn.functional.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def data_aug_mars():
    """
    Perform Data Augmentation for Mars Dataset

    This function performs data augmentation for the Mars dataset.
    It takes the original data, applies padding and sliding window
    technique to generate sequential data, and saves the augmented
    data and corresponding targets to files.

    """

    train_data = np.load('data/aicrowd1/train.npy', allow_pickle=True).item()

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
        # seq_list = np.moveaxis(seq_list, [4,0,1,2,3], [0,1,2,3,4])
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

    np.save('data/aicrowd1/train_seq_data.npy', data)
    np.save('data/aicrowd1/train_seq_target.npy', target)


def data_load_mars_seq(batch_size):  # pylint: disable = too-many-locals
    """
    Load Mars Sequential Data
    This function loads the Mars sequential data for training and validation.
    It performs data splitting, class sample count calculation, and creates
    data loaders for training and validation sets.

    Args:
        batch_size (int): The batch size for the data loaders.

    Returns:
        tuple: A tuple containing the training data loader, validation data
        loader, and class sample count.

    """

    data = np.load('data/train_seq_data.npy', allow_pickle=True)
    target = np.load('data/train_seq_target.npy', allow_pickle=True)

    shuf = np.random.permutation(data.shape[0])
    data = data[shuf]
    target = target[shuf]

    data_train, data_val = np.split(  # pylint: disable= unbalanced-tuple-unpacking
        data, [int(0.8 * data.shape[0])])
    target_train, target_val = np.split(  # pylint: disable= unbalanced-tuple-unpacking
        target, [int(0.8 * target.shape[0])])

    class_sample_count = []

    for class_type in np.unique(target_train):
        class_sample_count.append(len(np.where(target_train == class_type)[0]))

    class_sample_count = np.array(class_sample_count)

    weight = 1. / class_sample_count
    samples_weight = np.array([weight[target_train]])
    samples_weight = samples_weight.squeeze(0)
    sampler = torch.utils.data.WeightedRandomSampler(
        samples_weight, len(samples_weight))

    data_train = torch.Tensor(data_train)
    target_train = torch.Tensor(target_train)
    target_train = target_train.type(torch.long)  # pylint: disable= no-member

    data_val = torch.Tensor(data_val)
    target_val = torch.Tensor(target_val)
    target_val = target_val.type(torch.long)  # pylint: disable= no-member

    dataset_train = TensorDataset(data_train, target_train)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=sampler)

    dataset_val = TensorDataset(data_val, target_val)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True)

    return dataloader_train, dataloader_val, class_sample_count


def plot_perf(perf_data, strat, save_path='.'):  # pylint: disable= too-many-statements, too-many-locals
    """
    Plot performance metrics (loss, accuracy, F1 score, precision) for a specific model strategy.

    Args:
        perf_data (List): List containing the performance data for the model.
        strat (str): Model strategy ('cbfl', 'weighted_samp', 'none').
        save_path (str): local path location to save plots.

    Returns:
        None
    """

    perf_data = np.array(perf_data)
    train_losses, train_accs, train_cls_accs, train_f1s, train_precs, val_losses, val_accs, val_cls_accs, val_f1s, val_precs, random_accs, random_f1s, random_precs = perf_data  # pylint: disable=line-too-long
    title = 'TRANSFORMER'

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(['train', 'validation'], loc='best')
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.title(title)
    plt.savefig(save_path + '/plots/' + strat + '_model_loss.png')
    plt.close()
    plt.cla()
    plt.clf()

    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.plot(random_accs)
    plt.legend(['train', 'validation', 'random'], loc='best')
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    plt.title(title)
    plt.ylim(0, 1)
    plt.savefig(save_path + '/plots/' + strat + '_model_acc.png')
    plt.close()
    plt.cla()
    plt.clf()

    plt.plot(train_f1s)
    plt.plot(val_f1s)
    plt.plot(random_f1s)
    plt.legend(['train', 'validation', 'random'], loc='best')
    plt.xlabel('EPOCH')
    plt.ylabel('F1')
    plt.title(title)
    plt.ylim(0, 1)
    plt.savefig(save_path + '/plots/' + strat + '_model_f1.png')
    plt.close()
    plt.cla()
    plt.clf()

    plt.plot(train_precs)
    plt.plot(val_precs)
    plt.plot(random_precs)
    plt.legend(['train', 'validation', 'random'], loc='best')
    plt.xlabel('EPOCH')
    plt.ylabel('PRECISION')
    plt.title(title)
    plt.ylim(0, 1)
    plt.savefig(save_path + '/plots/' + strat + '_model_prec.png')
    plt.close()
    plt.cla()
    plt.clf()

    plt.plot([row[0] for row in train_cls_accs])
    plt.plot([row[1] for row in train_cls_accs])
    plt.plot([row[2] for row in train_cls_accs])
    plt.plot([row[3] for row in train_cls_accs])
    plt.legend(['class 0', 'class 1', 'class 2', 'class 3'], loc='best')
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY PER CLASS')
    plt.title(title + ' TRAIN')
    plt.ylim(0, 1)
    plt.savefig(save_path + '/plots/' + strat + '_train_per_class_acc.png')
    plt.close()
    plt.cla()
    plt.clf()

    plt.plot([row[0] for row in val_cls_accs])
    plt.plot([row[1] for row in val_cls_accs])
    plt.plot([row[2] for row in val_cls_accs])
    plt.plot([row[3] for row in val_cls_accs])
    plt.legend(['class 0', 'class 1', 'class 2', 'class 3'], loc='best')
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY PER CLASS')
    plt.title(title + ' VALIDATION')
    plt.ylim(0, 1)
    plt.savefig(save_path + '/plots/' + strat + '_val_per_class_acc.png')
    plt.close()
    plt.cla()
    plt.clf()


def random_perf(val_loader, num_classes):
    """
    Calculate performance metrics (accuracy, F1 score, precision) of a random prediction baseline
    for the given validation dataset.

    Args:
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        num_classes (int): Number of classes in the classification task.

    Returns:
        Tuple[float, float, float]: A tuple containing the following:
            - accs.mean (float): Mean accuracy of random predictions.
            - f1s.mean (float): Mean F1 score of random predictions.
            - precs.mean (float): Mean precision of random predictions.
    """

    accs = MeanMeter()
    f1s = MeanMeter()
    precs = MeanMeter()

    for _, (_, target) in enumerate(val_loader):

        preds = np.random.randint(0, num_classes, target.shape[0])

        acc = sklearn.metrics.accuracy_score(target, preds)
        f1_score = sklearn.metrics.f1_score(target, preds, average='macro')
        prec = sklearn.metrics.precision_score(
            target, preds, average='macro', zero_division=0)

        accs.update(acc, preds.shape[0])
        f1s.update(f1_score, preds.shape[0])
        precs.update(prec, preds.shape[0])

    return accs.mean, f1s.mean, precs.mean


def run(batch_size=None, epochs=None, learning_rate=None, weight_decay=None,  # pylint: disable= too-many-arguments, too-many-statements, too-many-locals
        save_path='.'):
    """
    Execute the complete training and evaluation pipeline.

    Args:
        batch_size (int): Batch size for training and validation.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimization.
        weight_decay (float): Weight decay for optimization.
        save_path (str): Local path where trained model is saved.

    Returns:
        Tuple[List[float], List[float], float]: A tuple containing the following:
            - val_f1s (List[float]): List of F1 scores for each validation epoch.
            - val_precs (List[float]): List of precision scores for each validation epoch.
            - run_time (float): Total execution time in seconds.
    """

    dataloader_train, dataloader_val, class_sample_count = data_load_mars_seq(
        batch_size)
    num_classes = len(class_sample_count)

    model = Transformer()
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    best_f1 = 0.0
    best_prec = 0.0
    best_cm = None
    best_model = None

    train_losses = []
    train_accs = []
    train_cls_accs = []
    train_f1s = []
    train_precs = []

    val_losses = []
    val_accs = []
    val_cls_accs = []
    val_f1s = []
    val_precs = []

    random_accs = []
    random_f1s = []
    random_precs = []

    start_time = time.time()

    for epoch in range(epochs):

        # train loop
        train_loss, train_acc, train_cm, train_f1, train_prec = train(
            epoch, dataloader_train, model, optimizer, criterion, num_classes)

        # validation loop
        val_loss, val_acc, val_cm, val_f1, val_prec = validate(
            dataloader_val, model, criterion, num_classes)

        # Random performance
        random_acc, random_f1, random_prec = random_perf(
            dataloader_val, num_classes)

        train_cls_acc = train_cm.diag().detach().numpy().tolist()
        val_cls_acc = val_cm.diag().detach().numpy().tolist()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_cls_accs.append(train_cls_acc)
        train_f1s.append(train_f1)
        train_precs.append(train_prec)

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_cls_accs.append(val_cls_acc)
        val_f1s.append(val_f1)
        val_precs.append(val_prec)

        random_accs.append(random_acc)
        random_f1s.append(random_f1)
        random_precs.append(random_prec)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_prec = val_prec
            best_cm = val_cm
            best_model = copy.deepcopy(model)

    print(f'Best F1: {best_f1:.4f}')
    print(f'Best Prec: {best_prec:.4f}')
    per_cls_acc = best_cm.diag().detach().numpy().tolist()

    for i, acc_i in enumerate(per_cls_acc):
        print(f"Best accuracy of class {i}: {acc_i:.4f}")

    print('')

    run_time = time.time() - start_time

    torch.save(
        best_model.state_dict(),
        save_path +
        '/checkpoints/transformer_model.pth')
    perf_data = [
        train_losses,
        train_accs,
        train_cls_accs,
        train_f1s,
        train_precs,
        val_losses,
        val_accs,
        val_cls_accs,
        val_f1s,
        val_precs,
        random_accs,
        random_f1s,
        random_precs]
    plot_perf(perf_data, 'transformer', save_path)

    return val_f1s, val_precs, run_time


def train(epoch, data_loader, model, optimizer, criterion, num_classes):  # pylint: disable = too-many-arguments, too-many-locals
    """
    Train the model for one epoch using the provided data loader, optimizer, and criterion.

    Args:
        epoch (int): Current epoch number.
        data_loader (torch.utils.data.DataLoader): Data loader for training data.
        model (torch.nn.Module): Model to be trained.
        optimizer: Optimizer used for training.
        criterion: Loss function used for training.
        num_classes (int): Number of classes in the classification task.

    Returns:
        Tuple[float, float, torch.Tensor, float, float]: A tuple containing the following:
            - losses.mean (float): Mean loss over the training set.
            - acc.mean (float): Mean accuracy over the training set.
            - correlation_matrix (torch.Tensor): Correlation matrix of predicted classes.
            - f1s.mean (float): Mean F1 score over the training set.
            - precs.mean (float): Mean precision over the training set.
    """
    iter_time = MeanMeter()
    losses = MeanMeter()
    acc = MeanMeter()
    f1s = MeanMeter()
    precs = MeanMeter()

    confusion_matrix = torch.zeros(num_classes, num_classes)  # pylint: disable= no-member

    for idx, (data, target) in enumerate(data_loader):

        data = data.permute((1, 0, 2))

        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        model.train()
        out = model(data)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calc accuracy
        _, preds = out.max(dim=-1)

        batch_acc = sklearn.metrics.accuracy_score(
            target.cpu().numpy(), preds.cpu().numpy())
        f1_score = sklearn.metrics.f1_score(
            target.cpu().numpy(),
            preds.cpu().numpy(),
            average='macro')
        prec = sklearn.metrics.precision_score(
            target.cpu().numpy(),
            preds.cpu().numpy(),
            average='macro', zero_division=0)

        # update confusion matrix
        for target_view, pred_view in zip(target.view(-1), preds.view(-1)):
            confusion_matrix[target_view.long(), pred_view.long()] += 1

        # Calc means
        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])
        f1s.update(f1_score, out.shape[0])
        precs.update(prec, out.shape[0])
        iter_time.update(time.time() - start)

        if idx % 100 == 0:
            print((
                'Epoch: [{0}][{1}/{2}]\t'  # pylint: disable= consider-using-f-string
                'Time {iter_time.val:.3f} ({iter_time.mean:.3f})\t'
                'Loss {loss.val:.4f} ({loss.mean:.4f})\t'
                'Accuracy {accu.val:.4f} ({accu.mean:.4f})\t').format(
                epoch, idx, len(data_loader),
                iter_time=iter_time, loss=losses, accu=acc))

    confusion_matrix = confusion_matrix / confusion_matrix.sum(1)

    return losses.mean.detach().cpu().numpy(), acc.mean, confusion_matrix, f1s.mean, precs.mean


def validate(val_loader, model, criterion, num_classes):  # pylint: disable = too-many-locals
    """
    Perform validation on the given model using the provided validation data loader.

    Args:
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        model (torch.nn.Module): Model to be evaluated.
        criterion: Loss function used for evaluation.
        num_classes (int): Number of classes in the classification task.

    Returns:
        Tuple[float, float, torch.Tensor, float, float]: A tuple containing the following:
            - losses.mean (float): Mean loss over the validation set.
            - acc.mean (float): Mean accuracy over the validation set.
            - correlation_matrix (torch.Tensor): Correlation matrix of predicted classes.
            - f1s.mean (float): Mean F1 score over the validation set.
            - precs.mean (float): Mean precision over the validation set.
    """

    iter_time = MeanMeter()
    losses = MeanMeter()
    acc = MeanMeter()
    f1s = MeanMeter()
    precs = MeanMeter()

    confusion_matrix = torch.zeros(num_classes, num_classes)  # pylint: disable=no-member

    # evaluation loop
    for _, (data, target) in enumerate(val_loader):
        start = time.time()

        data = data.permute((1, 0, 2))

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        model.eval()
        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)

        _, preds = out.max(dim=-1)
        batch_acc = sklearn.metrics.accuracy_score(
            target.cpu().numpy(), preds.cpu().numpy())
        f1_score = sklearn.metrics.f1_score(
            target.cpu().numpy(),
            preds.cpu().numpy(),
            average='macro')
        prec = sklearn.metrics.precision_score(
            target.cpu().numpy(),
            preds.cpu().numpy(),
            average='macro', zero_division=0)

        # update confusion matrix
        for target_view, pred_view in zip(target.view(-1), preds.view(-1)):
            confusion_matrix[target_view.long(), pred_view.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])
        f1s.update(f1_score, out.shape[0])
        precs.update(prec, out.shape[0])
        iter_time.update(time.time() - start)

    confusion_matrix = confusion_matrix / confusion_matrix.sum(1)
    per_cls_acc = confusion_matrix.diag().detach().numpy().tolist()

    print("")
    print("Validation Metrics:")

    for i, acc_i in enumerate(per_cls_acc):
        print(f"Accuracy mean of class {i}: {acc_i:.4f}")

    print(f"Accuracy mean total: {acc.mean:.4f}")
    print(f"Loss mean: {losses.mean:.4f}")
    print(f"F1 mean: {f1s.mean:.4f}")
    print(f"Prec mean: {precs.mean:.4f}")
    print("")

    return losses.mean.detach().cpu().numpy(), acc.mean, confusion_matrix, f1s.mean, precs.mean


def main():
    """main."""

    _, _, transformer_run_time = run(
        batch_size=64, epochs=10, learning_rate=0.001, weight_decay=0.001, save_path='.')

    print('')
    print('transformer_run_time: ' + str(transformer_run_time))
    print('')


if __name__ == '__main__':
    main()
