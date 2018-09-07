"""
Code from preprocessing the UCSD dataset.
"""
import os
import shutil
import zipfile
from urllib.request import urlretrieve

import imageio
import numpy as np
import scipy.io
from torch.utils.data import Dataset

from crowd.data import CrowdExampleWithPerspective
from crowd.label_generation import generate_density_label
from utility import download_and_extract_file, seed_all

if os.path.basename(os.path.normpath(os.path.abspath('..'))) == 'srgan':
    database_directory = '../../ShanghaiTech Dataset'
else:
    database_directory = '../ShanghaiTech Dataset'


class ShanghaiTechDataset(Dataset):
    """
    A class for the UCSD crowd dataset.
    """
    def __init__(self, dataset='train', transform=None, seed=None, unlabeled=False):
        seed_all(seed)
        dataset_directory = os.path.join(database_directory, 'part_B', '{}_data'.format(dataset))
        if unlabeled:
            self.images = np.load(os.path.join(dataset_directory, 'unlabeled_images.npy'))
            self.labels = np.zeros(self.images.shape[:3], dtype=np.float32)
        else:
            self.images = np.load(os.path.join(dataset_directory, 'images.npy'))
            self.labels = np.load(os.path.join(dataset_directory, 'labels.npy'))
        self.perspectives = np.zeros_like(self.labels)
        self.rois = np.ones_like(self.labels, dtype=np.bool)
        self.length = self.labels.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        """
        :param index: The index within the entire dataset.
        :type index: int
        :return: An example and label from the crowd dataset.
        :rtype: torch.Tensor, torch.Tensor
        """
        example = CrowdExampleWithPerspective(image=self.images[index], label=self.labels[index], roi=self.rois[index],
                                              perspective=self.perspectives[index])
        if self.transform:
            example = self.transform(example)
        return example.image, example.label

    def __len__(self):
        return self.length


class ShanghaiTechPreprocessing:
    """A class for preparing the ShanghaiTech dataset."""
    def download_and_preprocess(self):
        """Downloads and preprocesses the database."""
        print('Preparing ShanghaiTech database.')
        print('Downloading...')
        self.download()
        print('Preprocessing...')
        self.preprocess()

    @staticmethod
    def download():
        """Downloads the database."""
        if os.path.exists(database_directory):
            shutil.rmtree(database_directory)
        os.makedirs(database_directory)
        urlretrieve('https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=1', os.path.join(database_directory, 'temporary'))
        with zipfile.ZipFile(os.path.join(database_directory, 'temporary'), 'r') as zip_file:
            zip_file.extractall(database_directory)
        files = os.listdir(os.path.join(database_directory, 'ShanghaiTech'))
        for file_ in files:
            shutil.move(os.path.join(database_directory, 'ShanghaiTech', file_), database_directory)
        shutil.rmtree(os.path.join(database_directory, 'ShanghaiTech'))
        os.remove(os.path.join(database_directory, 'temporary'))

        print('Done downloading.')

    def preprocess(self):
        """Preprocesses the database to the format needed by the network."""
        for part in ['part_B']:
            for dataset in ['test_data', 'train_data']:
                image_list = []
                label_list = []
                ground_truth_directory = os.path.join(database_directory, part, dataset, 'ground-truth')
                image_directory = os.path.join(database_directory, part, dataset, 'images')
                for mat_filename in os.listdir(ground_truth_directory):
                    image_filename = mat_filename[3:-3]
                    mat_path = os.path.join(ground_truth_directory, mat_filename)
                    image_path = os.path.join(image_directory, image_filename + 'jpg')
                    image = imageio.imread(image_path)
                    label_size = image.shape[:2]
                    mat = scipy.io.loadmat(mat_path)
                    head_positions = mat['image_info'][0, 0][0][0][0]
                    label = generate_density_label(head_positions, label_size)
                    image_list.append(image)
                    label_list.append(label)
                images = np.stack(image_list)
                labels = np.stack(label_list)
                np.save(os.path.join(database_directory, part, dataset, 'images.npy'), images)
                np.save(os.path.join(database_directory, part, dataset, 'labels.npy'), labels)


class ShanghaiTechCheck:
    def display_statistics(self):
        print('=' * 50)
        print('part_B')
        test_dataset_name = 'test'
        test_dataset_directory = os.path.join(database_directory, 'part_B', '{}_data'.format(test_dataset_name))
        test_images = np.load(os.path.join(test_dataset_directory, 'images.npy'))
        test_labels = np.load(os.path.join(test_dataset_directory, 'labels.npy'))
        self.print_statistics(test_dataset_name, test_images, test_labels)
        train_dataset_name = 'train'
        train_dataset_directory = os.path.join(database_directory, 'part_B', '{}_data'.format(train_dataset_name))
        train_images = np.load(os.path.join(train_dataset_directory, 'images.npy'))
        train_labels = np.load(os.path.join(train_dataset_directory, 'labels.npy'))
        self.print_statistics(train_dataset_name, train_images, train_labels)
        total_dataset_name = 'total'
        total_images = np.concatenate([test_images, train_images], axis=0)
        total_labels = np.concatenate([test_labels, train_labels], axis=0)
        self.print_statistics(total_dataset_name, total_images, total_labels)

    def print_statistics(self, dataset, images, labels):
        print('-' * 50)
        print('Dataset: {}'.format(dataset))
        print('Images shape: {}'.format(images.shape))
        print('Labels shape: {}'.format(labels.shape))
        print('Person count: {}'.format(labels.sum()))
        print('Average count: {}'.format(labels.sum(axis=(1, 2)).mean(axis=0)))
        print('Max single image count: {}'.format(labels.sum(axis=(1, 2)).max(axis=0)))
        print('Min single image count: {}'.format(labels.sum(axis=(1, 2)).min(axis=0)))


if __name__ == '__main__':
    preprocessor = ShanghaiTechPreprocessing()
    # preprocessor.download_and_preprocess()
    preprocessor.preprocess()
    ShanghaiTechCheck().display_statistics()
