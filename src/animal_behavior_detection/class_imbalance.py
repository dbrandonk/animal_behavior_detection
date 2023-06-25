import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.metrics
import time
import torch
from torch.utils.data import TensorDataset, DataLoader


class FocalLoss(torch.nn.Module):

    def __init__(self, samples_per_cls, no_of_classes, beta, gamma):
        super(FocalLoss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.beta = beta
        self.gamma = gamma

    def forward(self, input, target):
        """
        :param input: input predictions
        :param target: labels
        :return: loss
        """

        loss = cb_loss(
            target,
            input,
            self.samples_per_cls,
            self.no_of_classes,
            'focal',
            self.beta,
            self.gamma)

        return loss


class MeanMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean = self.sum / self.count


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.fc1 = torch.nn.Linear(28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.fc4 = torch.nn.Linear(512, 4)

    def forward(self, x):

        outs = torch.flatten(x, 1)
        outs = torch.nn.functional.relu(self.fc1(outs))
        outs = torch.nn.functional.relu(self.fc2(outs))
        outs = torch.nn.functional.relu(self.fc3(outs))
        outs = self.fc4(outs)

        return outs


def cb_loss(labels, logits, samples_per_cls,
            no_of_classes, loss_type, beta, gamma):
    """
    Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = torch.nn.functional.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)

    return cb_loss


def data_loader_mars(batch_size, strat):

    train_data = np.load('data/train.npy', allow_pickle=True).item()

    data = None
    target = None

    for k, seq in train_data['sequences'].items():
        kp = seq['keypoints']
        annot = seq['annotations']

        if data is None:
            data = kp.copy()
            target = annot.copy()
        else:
            data = np.concatenate((data, kp), axis=0)
            target = np.concatenate((target, annot), axis=0)

    shuf = np.random.permutation(data.shape[0])
    data = data[shuf]
    target = target[shuf]

    data_train, data_val = np.split(data, [int(0.8 * data.shape[0])])
    target_train, target_val = np.split(target, [int(0.8 * target.shape[0])])

    class_sample_count = []

    for class_type in np.unique(target_train):
        class_sample_count.append(len(np.where(target_train == class_type)[0]))

    class_sample_count = np.array(class_sample_count)

    if strat == 'weighted_samp':
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[target_train]])
        samples_weight = samples_weight.squeeze(0)
        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight, len(samples_weight))

    data_train = torch.Tensor(data_train)
    target_train = torch.Tensor(target_train)
    target_train = target_train.type(torch.long)

    data_val = torch.Tensor(data_val)
    target_val = torch.Tensor(target_val)
    target_val = target_val.type(torch.long)

    dataset_train = TensorDataset(data_train, target_train)

    if strat == 'weighted_samp':
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=sampler)
    else:
        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = TensorDataset(data_val, target_val)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True)

    return dataloader_train, dataloader_val, class_sample_count


def focal_loss(labels, logits, alpha, gamma):
    """
    Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """

    BCLoss = torch.nn.functional.binary_cross_entropy_with_logits(
        input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(
            1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


def plot_perf(perf_data, strat):

    perf_data = np.array(perf_data)
    train_losses, train_accs, train_cls_accs, train_f1s, train_precs, val_losses, val_accs, val_cls_accs, val_f1s, val_precs, random_accs, random_f1s, random_precs = perf_data

    if (strat == 'cbfl'):
        title = 'CLASS BALANCED FOCAL LOSS'
    elif (strat == 'weighted_samp'):
        title = 'WEIGHTED SAMPLING'
    elif (strat == 'none'):
        title = 'NONE'

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(['train', 'validation'], loc='best')
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.title(title)
    plt.savefig('./plots/' + strat + '_model_loss.png')
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
    plt.savefig('./plots/' + strat + '_model_acc.png')
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
    plt.savefig('./plots/' + strat + '_model_f1.png')
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
    plt.savefig('./plots/' + strat + '_model_prec.png')
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
    plt.savefig('./plots/' + strat + '_train_per_class_acc.png')
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
    plt.savefig('./plots/' + strat + '_val_per_class_acc.png')
    plt.close()
    plt.cla()
    plt.clf()


def plot_perf_comp(perf_data):

    perf_data = np.array(perf_data)
    cbfl_f1s, cbfl_precs, cbfl_run_time, ws_f1s, ws_precs, ws_run_time, none_f1s, none_precs, none_run_time, random_f1s, random_precs = perf_data

    plt.plot(cbfl_f1s)
    plt.plot(ws_f1s)
    plt.plot(none_f1s)
    plt.plot(random_f1s)
    plt.legend(['CBFL', 'WEIGHTED', 'NONE', 'RANDOM'], loc='best')
    plt.xlabel('EPOCH')
    plt.ylabel('F1')
    plt.title('F1 COMPARISON')
    plt.ylim(0, 1)
    plt.savefig('./plots/comp_model_f1.png')
    plt.close()
    plt.cla()
    plt.clf()

    plt.plot(cbfl_precs)
    plt.plot(ws_precs)
    plt.plot(none_precs)
    plt.plot(random_precs)
    plt.legend(['CBFL', 'WEIGHTED', 'NONE', 'RANDOM'], loc='best')
    plt.xlabel('EPOCH')
    plt.ylabel('PRECISION')
    plt.title('PRECISION COMPARISON')
    plt.ylim(0, 1)
    plt.savefig('./plots/comp_model_prec.png')
    plt.close()
    plt.cla()
    plt.clf()


def random_perf(val_loader, num_classes):

    accs = MeanMeter()
    f1s = MeanMeter()
    precs = MeanMeter()

    for idx, (data, target) in enumerate(val_loader):

        preds = np.random.randint(0, num_classes, target.shape[0])

        acc = sklearn.metrics.accuracy_score(target, preds)
        f1 = sklearn.metrics.f1_score(target, preds, average='macro')
        prec = sklearn.metrics.precision_score(
            target, preds, average='macro', zero_division=0)

        accs.update(acc, preds.shape[0])
        f1s.update(f1, preds.shape[0])
        precs.update(prec, preds.shape[0])

    return accs.mean, f1s.mean, precs.mean


def run(batch_size=None, epochs=None, learning_rate=None,
        weight_decay=None, momentum=None, beta=None, gamma=None, strat=None):

    dataloader_train, dataloader_val, class_sample_count = data_loader_mars(
        batch_size,
        strat)
    num_classes = len(class_sample_count)

    model = Network()
    if torch.cuda.is_available():
        model = model.cuda()

    if strat == 'cbfl':
        criterion = FocalLoss(class_sample_count, num_classes, beta, gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
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
            epoch, dataloader_val, model, criterion, num_classes)

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

    print('Best F1: {:.4f}'.format(best_f1))
    print('Best Prec: {:.4f}'.format(best_prec))
    per_cls_acc = best_cm.diag().detach().numpy().tolist()

    for i, acc_i in enumerate(per_cls_acc):
        print("Best accuracy of class {}: {:.4f}".format(i, acc_i))

    print('')

    run_time = time.time() - start_time

    torch.save(
        best_model.state_dict(),
        './checkpoints/' +
        strat +
        '_model.pth')
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
    plot_perf(perf_data, strat)

    return val_f1s, val_precs, run_time


def train(epoch, data_loader, model, optimizer, criterion, num_classes):

    iter_time = MeanMeter()
    losses = MeanMeter()
    acc = MeanMeter()
    f1s = MeanMeter()
    precs = MeanMeter()

    cm = torch.zeros(num_classes, num_classes)

    for idx, (data, target) in enumerate(data_loader):

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

        value, preds = out.max(dim=-1)
        batch_acc = sklearn.metrics.accuracy_score(target, preds)
        f1 = sklearn.metrics.f1_score(target, preds, average='macro')
        prec = sklearn.metrics.precision_score(
            target, preds, average='macro', zero_division=0)

        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])
        f1s.update(f1, out.shape[0])
        precs.update(prec, out.shape[0])
        iter_time.update(time.time() - start)

        if idx % 100 == 0:
            print((
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {iter_time.val:.3f} ({iter_time.mean:.3f})\t'
                'Loss {loss.val:.4f} ({loss.mean:.4f})\t'
                'Accuracy {accu.val:.4f} ({accu.mean:.4f})\t').format(
                epoch, idx, len(data_loader),
                iter_time=iter_time, loss=losses, accu=acc))

    cm = cm / cm.sum(1)

    return losses.mean.detach().numpy(), acc.mean, cm, f1s.mean, precs.mean


def validate(epoch, val_loader, model, criterion, num_classes):

    iter_time = MeanMeter()
    losses = MeanMeter()
    acc = MeanMeter()
    f1s = MeanMeter()
    precs = MeanMeter()

    cm = torch.zeros(num_classes, num_classes)

    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        model.eval()
        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)

        value, preds = out.max(dim=-1)
        batch_acc = sklearn.metrics.accuracy_score(target, preds)
        f1 = sklearn.metrics.f1_score(target, preds, average='macro')
        prec = sklearn.metrics.precision_score(
            target, preds, average='macro', zero_division=0)

        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])
        f1s.update(f1, out.shape[0])
        precs.update(prec, out.shape[0])
        iter_time.update(time.time() - start)

    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()

    print("")
    print("Validation Metrics:")

    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy mean of class {}: {:.4f}".format(i, acc_i))

    print(("Accuracy mean total: {accuracy.mean:.4f}").format(accuracy=acc))
    print(("Loss mean: {loss.mean:.4f}").format(loss=losses))
    print(("F1 mean: {F1.mean:.4f}").format(F1=f1s))
    print(("Prec mean: {PREC.mean:.4f}").format(PREC=precs))
    print("")

    return losses.mean.detach().numpy(), acc.mean, cm, f1s.mean, precs.mean


def main():

    cbfl_f1s, cbfl_precs, cbfl_run_time = run(
        batch_size=64, epochs=10, learning_rate=0.00001, weight_decay=0.001,
        momentum=0.9, beta=0.9999, gamma=1.0, strat='cbfl')
    ws_f1s, ws_precs, ws_run_time = run(
        batch_size=64, epochs=10, learning_rate=0.00001, weight_decay=0.001,
        momentum=0.9, beta=0.9999, gamma=1.0, strat='weighted_samp')
    none_f1s, none_precs, none_run_time = run(
        batch_size=64, epochs=10, learning_rate=0.00001, weight_decay=0.001,
        momentum=0.9, beta=0.9999, gamma=1.0, strat='none')

    dataloader_train, dataloader_val, class_sample_count = data_loader_mars(
        64, 'none')
    num_classes = len(class_sample_count)

    random_accs = []
    random_f1s = []
    random_precs = []

    for epoch in range(10):
        random_acc, random_f1, random_prec = random_perf(
            dataloader_val, num_classes)
        random_accs.append(random_acc)
        random_f1s.append(random_f1)
        random_precs.append(random_prec)

    print('')
    print('cbfl_run_time: ' + str(cbfl_run_time))
    print('ws_run_time: ' + str(ws_run_time))
    print('none_run_time: ' + str(none_run_time))
    print('')

    perf_data = [
        cbfl_f1s,
        cbfl_precs,
        cbfl_run_time,
        ws_f1s,
        ws_precs,
        ws_run_time,
        none_f1s,
        none_precs,
        none_run_time,
        random_f1s,
        random_precs]
    plot_perf_comp(perf_data)


if __name__ == '__main__':
    main()
