from keras.utils import to_categorical
from keras.preprocessing.text import one_hot
import numpy as np
import os
from random import shuffle
from clusterone import get_data_path


class Data(object):
    DATASET_PATH = "/Users/abhishekpradhan/Workspace/Datasets/"

    def __init__(self, chars, seq_len, batch_size=50):
        self.data_path = get_data_path(
            dataset_name="abhishek/aclimdb",
            local_root=self.DATASET_PATH,
            local_repo="aclImdb",
            path=''
        )

        self.chars = chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.char_vocab, self.onehot_matirx = self.get_char_onehot()

    def load_data(self, train):
        pos_path = ""
        neg_path = ""
        if train:
            pos_path = os.path.join(self.data_path, "train/pos")
            neg_path = os.path.join(self.data_path, "train/neg")
        else:
            pos_path = os.path.join(self.data_path, "test/pos")
            neg_path = os.path.join(self.data_path, "test/neg")

        print("Loading Datasets.....")
        pos_lines = self.read_files(pos_path)
        neg_lines = self.read_files(neg_path)
        return pos_lines, neg_lines

    def read_files(self, dir_path):
        print("Reading dataset from " + dir_path)
        lines = []
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            line = open(file_path, 'r', encoding='utf-8').read()
            lines.append(line)
        return lines

    def get_char_onehot(self):
        print("Creating One-Hot Matrix.....")
        char_list = list(self.chars)
        chars = {ch: i for i, ch in enumerate(char_list, start=1)}
        print("Total number of charaters in Vocab : " + str(len(char_list)))
        onehot_matirx = np.identity(len(char_list))
        zeros = np.zeros((1, len(self.chars)))
        onehot_matirx = np.concatenate((zeros, onehot_matirx))
        return chars, onehot_matirx

    def encode_dataset(self, train):
        pos, neg = self.load_data(train)
        texts = pos + neg
        labels = [1 for _ in pos] + [0 for _ in neg]
        zipped_datset = list(zip(texts, labels))
        shuffle(zipped_datset)
        print("Encoding dataset ..... ")
        encode_matrix = np.zeros((len(texts), self.seq_len, len(self.chars)))
        for i in range(len(zipped_datset)):
            text, _ = zipped_datset[i]
            encode_matrix[i] = self.encode_text(text)
        return encode_matrix, np.array(labels)

    def encode_text(self, text):
        encoded = np.zeros((self.seq_len, len(self.chars)))
        for i in range(min(self.seq_len, len(text))):
            char = text[i].lower()
            if char in self.char_vocab:
                encoded[i] = self.onehot_matirx[self.char_vocab[char]]
            # else:
            #     encoded[i] = self.onehot_matirx[0]
        return encoded

    def get_batches(self, train=True):
        encode_matrix, labels = self.encode_dataset(train)
        batch_num = encode_matrix.shape[0] / self.batch_size
        print("Creating Batches....")
        print("No. of Batches : " + str(batch_num))
        text_batches = []
        label_batches = []
        for i in range(int(batch_num)):
            start = i * self.batch_size
            end = start + self.batch_size
            text_batch = encode_matrix[start: end, :, :]
            label_batch = labels[start: end]
            text_batches.append(text_batch)
            label_batches.append(label_batch)
        return text_batches, label_batches


# # Data Params
# chars = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
# seq_len = 1014
# batch_size = 100
# data = Data(chars, seq_len)
