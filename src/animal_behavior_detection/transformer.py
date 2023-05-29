import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import time
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import TensorDataset, DataLoader

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

class PositionalEncoding(torch.nn.Module):

    """
    PositionalEncoding module injects some information about the
    relative or absolute position of the tokens in the sequence. The
    positional encodings have the same dimension as the features so that
    the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
    different frequencies.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, feature]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(torch.nn.Module):

    def __init__(self):
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
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):

        out = self.pos_encoder(x)
        out = self.transformer_encoder(out)
        out = out.permute((1, 0, 2))
        out = torch.flatten(out, 1)
        out = torch.nn.functional.relu(self.fc1(out))
        out = self.fc2(out)
        return out

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

def data_load_mars_seq(batch_size):

    data = np.load('data/aicrowd1/train_seq_data.npy',allow_pickle=True)
    target = np.load('data/aicrowd1/train_seq_target.npy',allow_pickle=True)

    shuf = np.random.permutation(data.shape[0])
    data = data[shuf]
    target = target[shuf]

    data_train, data_val = np.split(data, [int(0.8 * data.shape[0])])
    target_train, target_val = np.split(target, [int(0.8 * target.shape[0])])

    class_sample_count = []

    for class_type in np.unique(target_train):
        class_sample_count.append(len(np.where(target_train == class_type)[0]))

    class_sample_count = np.array(class_sample_count)

    weight = 1. / class_sample_count
    samples_weight = np.array([weight[target_train]])
    samples_weight = samples_weight.squeeze(0)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    data_train = torch.Tensor(data_train)
    target_train = torch.Tensor(target_train)
    target_train = target_train.type(torch.long)

    data_val = torch.Tensor(data_val)
    target_val = torch.Tensor(target_val)
    target_val = target_val.type(torch.long)

    dataset_train = TensorDataset(data_train, target_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler)

    dataset_val = TensorDataset(data_val, target_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    return dataloader_train, dataloader_val, class_sample_count

def plot_perf(perf_data, strat, save_path='.'):

    perf_data = np.array(perf_data)
    train_losses, train_accs, train_cls_accs, train_f1s, train_precs, val_losses, val_accs, val_cls_accs, val_f1s, val_precs, random_accs, random_f1s, random_precs = perf_data
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

    accs = MeanMeter()
    f1s = MeanMeter()
    precs = MeanMeter()

    for idx, (data, target) in enumerate(val_loader):

        preds = np.random.randint(0, num_classes, target.shape[0])

        acc = sklearn.metrics.accuracy_score(target, preds)
        f1 = sklearn.metrics.f1_score(target, preds, average='macro')
        prec = sklearn.metrics.precision_score(target, preds, average='macro', zero_division=0)

        accs.update(acc, preds.shape[0])
        f1s.update(f1, preds.shape[0])
        precs.update(prec, preds.shape[0])

    return accs.mean, f1s.mean, precs.mean

def run(batch_size=None, epochs=None, learning_rate=None, weight_decay=None, momentum=None, beta=None, gamma=None, strat=None, save_path='.'):

    dataloader_train, dataloader_val, class_sample_count = data_load_mars_seq(batch_size)
    num_classes = len(class_sample_count)

    model = Transformer()
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
        train_loss, train_acc, train_cm, train_f1, train_prec  = train(epoch, dataloader_train, model, optimizer, criterion, num_classes)

        # validation loop
        val_loss, val_acc, val_cm, val_f1, val_prec = validate(epoch, dataloader_val, model, criterion, num_classes)

        # Random performance
        random_acc, random_f1, random_prec = random_perf(dataloader_val, num_classes)

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

    torch.save(best_model.state_dict(), save_path + '/checkpoints/transformer_model.pth')
    perf_data = [train_losses, train_accs, train_cls_accs, train_f1s, train_precs, val_losses, val_accs, val_cls_accs, val_f1s, val_precs, random_accs, random_f1s, random_precs]
    plot_perf(perf_data, 'transformer', save_path)

    return val_f1s, val_precs, run_time

def train(epoch, data_loader, model, optimizer, criterion, num_classes):

    iter_time = MeanMeter()
    losses = MeanMeter()
    acc = MeanMeter()
    f1s = MeanMeter()
    precs = MeanMeter()

    cm = torch.zeros(num_classes, num_classes)

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
        value, preds = out.max(dim=-1)

        batch_acc = sklearn.metrics.accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
        f1 = sklearn.metrics.f1_score(target.cpu().numpy(), preds.cpu().numpy(), average='macro')
        prec = sklearn.metrics.precision_score(target.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)

        # update confusion matrix
        for t, p in zip(target.view(-1), preds.view(-1)):
          cm[t.long(), p.long()] += 1

        # Calc means
        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])
        f1s.update(f1, out.shape[0])
        precs.update(prec, out.shape[0])
        iter_time.update(time.time() - start)

        if idx % 100 == 0:
          print(('Epoch: [{0}][{1}/{2}]\t'
                  'Time {iter_time.val:.3f} ({iter_time.mean:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.mean:.4f})\t'
                  'Accuracy {accu.val:.4f} ({accu.mean:.4f})\t')
                .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, accu=acc))

    cm = cm / cm.sum(1)

    return losses.mean.detach().cpu().numpy(), acc.mean, cm, f1s.mean, precs.mean

def validate(epoch, val_loader, model, criterion, num_classes):

  iter_time = MeanMeter()
  losses = MeanMeter()
  acc = MeanMeter()
  f1s = MeanMeter()
  precs = MeanMeter()

  cm = torch.zeros(num_classes, num_classes)

  # evaluation loop
  for idx, (data, target) in enumerate(val_loader):
    start = time.time()

    data = data.permute((1, 0, 2))

    if torch.cuda.is_available():
      data = data.cuda()
      target = target.cuda()

    model.eval()
    with torch.no_grad():
      out = model(data)
      loss = criterion(out, target)

    value, preds = out.max(dim=-1)
    batch_acc = sklearn.metrics.accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
    f1 = sklearn.metrics.f1_score(target.cpu().numpy(), preds.cpu().numpy(), average='macro')
    prec = sklearn.metrics.precision_score(target.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)

    # update confusion matrix
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

  return losses.mean.detach().cpu().numpy(), acc.mean, cm, f1s.mean, precs.mean

def main():

    transformer_f1s, transformer_precs, transformer_run_time = run(batch_size = 64, epochs=10, learning_rate = 0.001, weight_decay=0.001, momentum=0.9, beta=0.9999, gamma=1.0, strat=None, save_path='/content/drive/MyDrive/SCHOOL/CS7643/CS7643-PROJECT-SHARED/')

    print('')
    print('transformer_run_time: ' + str(transformer_run_time))
    print('')

if __name__ == '__main__':
    main()
