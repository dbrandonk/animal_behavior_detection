from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

def data_sample_count_mars():

    train_data = np.load('data/aicrowd1/train.npy',allow_pickle=True).item()

    data = None
    target = None

    for k, seq in train_data['sequences'].items():
        kp = seq['keypoints']
        annot = seq['annotations']

        if data is None:
            data = kp.copy()
            target = annot.copy()
        else:
            data = np.concatenate((data, kp), axis = 0)
            target = np.concatenate((target, annot), axis=0)


    class_sample_count = []

    for class_type in np.unique(target):
        class_sample_count.append(len(np.where(target == class_type)[0]))

    class_sample_count = np.array(class_sample_count)

    return class_sample_count

def num_to_text(anno_list):

    train = np.load('data/aicrowd1/train.npy',allow_pickle=True).item()
    number_to_class = {i: s for i, s in enumerate(train['vocabulary'])}
    return np.vectorize(number_to_class.get)(anno_list)

def plot_annotation_strip(annotation_sequence, start_frame = 0, stop_frame = 100, title="Behavior Labels"):

  train = np.load('data/aicrowd1/train.npy',allow_pickle=True).item()
  class_to_color = {'other': 'white', 'attack' : 'red', 'mount' : 'green', 'investigation': 'orange'}
  class_to_number = {s: i for i, s in enumerate(train['vocabulary'])}
  number_to_class = {i: s for i, s in enumerate(train['vocabulary'])}

  annotation_num = []
  for item in annotation_sequence[start_frame:stop_frame]:
    annotation_num.append(class_to_number[item])

  all_classes = list(set(annotation_sequence[start_frame:stop_frame]))

  cmap = colors.ListedColormap(['red', 'orange', 'green', 'white'])
  bounds=[-0.5,0.5,1.5, 2.5, 3.5]
  norm = colors.BoundaryNorm(bounds, cmap.N)

  height = 200
  arr_to_plot = np.repeat(np.array(annotation_num)[:,np.newaxis].transpose(),
                                                  height, axis = 0)

  fig, ax = plt.subplots(figsize = (16, 3))
  ax.imshow(arr_to_plot, interpolation = 'none',cmap=cmap, norm=norm)

  ax.set_yticks([])
  ax.set_xlabel('Frame Number')
  plt.title(title)

  import matplotlib.patches as mpatches

  legend_patches = []
  for item in all_classes:
    legend_patches.append(mpatches.Patch(color=class_to_color[item], label=item))

  plt.legend(handles=legend_patches,loc='center left', bbox_to_anchor=(1, 0.5))

  plt.tight_layout()

  plt.savefig('./plots/class_imbalance_strip.png')
  plt.close()
  plt.cla()
  plt.clf()

def plot_video():

    train = np.load('data/aicrowd1/train.npy',allow_pickle=True).item()

    for k, seq in train['sequences'].items():
        kp = seq['keypoints']
        annot = seq['annotations']

        text_sequence = num_to_text(annot)

        plot_annotation_strip( text_sequence, start_frame=0, stop_frame=len(annot) + 1000 )
        break

def main():

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
