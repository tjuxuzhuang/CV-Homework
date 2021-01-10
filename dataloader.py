import numpy as np
import struct
import os


class DataLoader:
    def __init__(self, dataset, batch_size, normalization, shuffle=True, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        try:
            self.data = np.load(self.dataset + r'\data.npy')
            self.label = np.load(self.dataset + r'\label.npy')
        except IOError:
            print("Error: 没有找到文件或读取文件失败")
            exit()
        else:
            print("文件读取成功")
        if normalization:
            self.normalization()
        if shuffle:
            self.shuffle()
        self.batch_num = int(np.floor(len(self.data) / self.batch_size))
        self.data_queue = np.array(np.array_split(self.data, self.batch_num))
        self.label_queue = np.array(np.array_split(self.label, self.batch_num))

    def shuffle(self):
        if self.seed:
            np.random.seed(self.seed)
            np.random.shuffle(self.data)
            np.random.seed(self.seed)
            np.random.shuffle(self.label)
        else:
            seed = np.random.randint(1, 100)
            np.random.seed(seed)
            np.random.shuffle(self.data)
            np.random.seed(seed)
            np.random.shuffle(self.label)

    def normalization(self):
        data_range = np.max(self.data) - np.min(self.data)
        self.data = (self.data - np.min(self.data)) / data_range
        label_range = np.max(self.label) - np.min(self.label)
        self.label = (self.label - np.min(self.label)) / label_range


class MNISTLoader:
    def __init__(self, dataset, kind, classes, batch_size, shuffle=True, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.kind = kind
        self.seed = seed
        self.classes = classes
        labels_path = os.path.join(self.dataset + str(self.kind) + '-labels.idx1-ubyte')
        images_path = os.path.join(self.dataset + str(self.kind) + '-images.idx3-ubyte')

        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            self.labels = np.fromfile(lbpath, dtype=np.uint8).reshape(-1, 1)
            self.labels = self.get_one_hot(self.labels)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            self.images = np.fromfile(imgpath, dtype=np.uint8).reshape(-1, 784)

        if shuffle:
            self.shuffle()
        self.batch_num = int(np.floor(len(self.images) / self.batch_size))
        self.data_queue = np.array(np.array_split(self.images, self.batch_num))
        self.label_queue = np.array(np.array_split(self.labels, self.batch_num))

    def shuffle(self):
        if self.seed:
            np.random.seed(self.seed)
            np.random.shuffle(self.images)
            np.random.seed(self.seed)
            np.random.shuffle(self.labels)
        else:
            seed = np.random.randint(1, 100)
            np.random.seed(seed)
            np.random.shuffle(self.images)
            np.random.seed(seed)
            np.random.shuffle(self.labels)

    def get_one_hot(self, label):
            # 对label进行one_hot编码
            one_hot = np.zeros([label.shape[0], self.classes])
            for i in range(label.shape[0]):
                one_hot[i][label[i]] = 1
            return one_hot
