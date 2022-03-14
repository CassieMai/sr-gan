"""
Code from preprocessing the clustered-orange dataset.
"""
import os
import random
import imageio
import numpy as np
import scipy.io
import torchvision
from torch.utils.data import Dataset

from crowd.data import CrowdExample, ExtractPatchForPosition, NegativeOneToOneNormalizeImage, NumpyArraysToTorchTensors
from crowd.database_preprocessor import DatabasePreprocessor

from utility import seed_all
from PIL import Image
import pandas as pd
import cv2


class ClusteredOrangeFullImageDataset(Dataset):
    """
    A class for the UCF QNRF full image crowd dataset.
    """
    def __init__(self, dataset='train', seed=None, number_of_examples=None, map_directory_name='maps',
                 examples_start=None):
        seed_all(seed)
        if examples_start is None:
            examples_end = number_of_examples
        elif number_of_examples is None:
            examples_end = None
        else:
            examples_end = examples_start + number_of_examples
        seed_all(seed)
        self.dataset_directory = os.path.join(ClusteredOrangePreprocessor().database_directory, dataset)
        file_names = os.listdir(self.dataset_directory + '_den')
        random.shuffle(file_names)
        self.file_names = [name for name in file_names if name.endswith('.csv')][examples_start:examples_end]
        self.length = len(self.file_names)
        self.map_directory_name = map_directory_name

    def __getitem__(self, index):
        """
        :param index: The index within the entire dataset.
        :type index: int
        :return: An example and label from the crowd dataset.
        :rtype: torch.Tensor, torch.Tensor
        """
        scale = 0.6
        file_name = self.file_names[index]
        image = Image.open(os.path.join(self.dataset_directory, file_name[:-4] + '.jpg')).convert('RGB')
        image = np.array(image)
        image = cv2.resize(image, (int(np.ceil(image.shape[1]*scale/8.0)*8), int(np.ceil(image.shape[0]*scale/8.0)*8)), interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float32, copy=False)
        density = pd.read_csv(os.path.join(self.dataset_directory + self.map_directory_name, file_name[:-4] + '.csv'),
                              sep=',', header=None).values
        density = density.astype(np.float32, copy=False)
        h = density.shape[0]
        w = density.shape[1]
        density = cv2.resize(density, (int(np.ceil(w*scale/8.0)*8), int(np.ceil(h*scale/8.0)*8)), interpolation=cv2.INTER_CUBIC)
        density = density * ((w * h) / (int(np.ceil(w*scale/8.0)*8) * int(np.ceil(h*scale/8.0)*8)))     
        label = density
        image = np.transpose(image, (2, 0, 1))
        # density = np.transpose(density, (2, 0, 1))
        print('image', image.shape)
        print('density', density.shape)
        return image, label, density

    def __len__(self):
        return self.length


class ClusteredOrangeTransformedDataset(Dataset):
    """
    A class for the transformed clustered orange crowd dataset.
    """
    def __init__(self, dataset='train', image_patch_size=224, label_patch_size=224, seed=None, number_of_examples=None,
                 middle_transform=None, map_directory_name='maps', examples_start=None):
        seed_all(seed)
        if examples_start is None:
            examples_end = number_of_examples
        elif number_of_examples is None:
            examples_end = None
        else:
            examples_end = examples_start + number_of_examples
        self.dataset_directory = os.path.join(ClusteredOrangePreprocessor().database_directory, dataset)
        print('dataset_directory', self.dataset_directory)
        file_names = os.listdir(self.dataset_directory) # + map_directory_name)
        random.shuffle(file_names)
        # self.file_names = [name for name in file_names if name.endswith('.csv')][examples_start:examples_end]
        self.file_names = [name for name in file_names if name.endswith('.jpg')][examples_start:examples_end]
        self.image_patch_size = image_patch_size
        self.label_patch_size = label_patch_size
        half_patch_size = int(self.image_patch_size // 2)
        self.length = 0
        self.start_indexes = []
        print('self.file_names', len(self.file_names))
        print('self.dataset_directory', self.dataset_directory)
        for file_name in self.file_names:
            self.start_indexes.append(self.length)
            image = Image.open(os.path.join(self.dataset_directory, file_name[:-4]+'.jpg')).convert('RGB')
            image = np.array(image)
            y_positions = range(half_patch_size, image.shape[0] - half_patch_size + 1)
            x_positions = range(half_patch_size, image.shape[1] - half_patch_size + 1)
            image_indexes_length = len(y_positions) * len(x_positions)
            self.length += image_indexes_length
        self.middle_transform = middle_transform
        self.map_directory_name = map_directory_name

    def __getitem__(self, index):
        """
        :param index: The index within the entire dataset.
        :type index: int
        :return: An example and label from the crowd dataset.
        :rtype: torch.Tensor, torch.Tensor
        """
        scale = 0.8
        index_ = random.randrange(self.length)
        file_name_index = np.searchsorted(self.start_indexes, index_, side='right') - 1
        file_name = self.file_names[file_name_index]
        start_index = self.start_indexes[file_name_index]
        position_index = index_ - start_index
        extract_patch_transform = ExtractPatchForPosition(self.image_patch_size, self.label_patch_size,
                                                          allow_padded=True)  # In case image is smaller than patch.
        preprocess_transform = torchvision.transforms.Compose([NegativeOneToOneNormalizeImage(),
                                                               NumpyArraysToTorchTensors()])
        image = Image.open(os.path.join(self.dataset_directory, file_name[:-4] + '.jpg')).convert('RGB')
        image = np.array(image)
        image = cv2.resize(image, (int(np.ceil(image.shape[1]*scale/8.0)*8), int(np.ceil(image.shape[0]*scale/8.0)*8)), interpolation=cv2.INTER_CUBIC)
        density = pd.read_csv(os.path.join(self.dataset_directory + self.map_directory_name, file_name[:-4]+'.csv'), sep=',', header=None).values
        density = density.astype(np.float32, copy=False)
        h = density.shape[0]
        w = density.shape[1]
        density = cv2.resize(density, (int(np.ceil(w*scale/8.0)*8), int(np.ceil(h*scale/8.0)*8)), interpolation=cv2.INTER_CUBIC)
        density = density * ((w * h) / (int(np.ceil(w*scale/8.0)*8) * int(np.ceil(h*scale/8.0)*8)))
        label = density
        half_patch_size = int(self.image_patch_size // 2)
        y_positions = range(half_patch_size, image.shape[0] - half_patch_size + 1)
        x_positions = range(half_patch_size, image.shape[1] - half_patch_size + 1)
        positions_shape = [len(y_positions), len(x_positions)]
        print('position_index', position_index)
        print('positions_shape', positions_shape)
        y_index, x_index = np.unravel_index(position_index, positions_shape)
        y = y_positions[y_index]
        x = x_positions[x_index]
        example = CrowdExample(image=image, label=label, map_=density)

        example = extract_patch_transform(example, y, x)

        if self.middle_transform:
            example = self.middle_transform(example)
        example = preprocess_transform(example)
        map_ = example.map

        # visualization
        from utility import visualize
        print('image', image.shape)
        print('density', density.shape)
        visualize(image, density)

        return example.image, example.label, map_

    def __len__(self):
        return self.length


class ClusteredOrangePreprocessor(DatabasePreprocessor):
    """The preprocessor for the ClusteredOrange dataset."""
    def __init__(self):
        super().__init__()
        self.database_name = '/home/xiaocmai/scratch/datasets/colorization/debugset'   # ======  experimentset, debugset
        # self.database_url = 'http://crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip'
        self.database_archived_directory_name = 'xxxx'

    def preprocess(self):
        """Preprocesses the database generating the image and map labels."""
        for dataset_name_ in ['train', 'test']:
            dataset_directory = os.path.join(self.database_directory, dataset_name_)
            for mat_filename in os.listdir(os.path.join(self.database_directory, dataset_name_)):
                if not mat_filename.endswith('.mat'):
                    continue
                file_name = mat_filename[:-8]  # 8 for `_ann.mat` characters
                mat_path = os.path.join(self.database_directory, dataset_name_, mat_filename)
                original_image_path = os.path.join(self.database_directory, dataset_name_, file_name + '.jpg')
                image = imageio.imread(original_image_path)
                mat = scipy.io.loadmat(mat_path)
                original_head_positions = mat['annPoints']  # x, y ordering (mostly).
                # Get y, x ordering.
                head_positions = self.get_y_x_head_positions(original_head_positions, file_name,
                                                             label_size=image.shape[:2])
                self.generate_labels_for_example(dataset_directory, file_name, image, head_positions)

    @staticmethod
    def get_y_x_head_positions(original_head_positions, file_name, label_size):
        """Swaps the x's and y's of the head positions. Accounts for files where the labeling is incorrect."""
        if file_name == 'img_0087':
            # Flip y labels.
            head_position_list = []
            for original_head_position in original_head_positions:
                head_position_list.append([label_size[0] - original_head_position[0], original_head_position[1]])
            head_positions = np.array(head_position_list)
            return head_positions
        elif file_name == 'img_0006':
            # Flip x labels.
            head_position_list = []
            for original_head_position in original_head_positions:
                head_position_list.append([original_head_position[0], label_size[1] - original_head_position[1]])
            head_positions = np.array(head_position_list)
            return head_positions
        else:
            return original_head_positions[:, [1, 0]]


if __name__ == '__main__':
    preprocessor = ClusteredOrangePreprocessor()
    preprocessor.download_and_preprocess()
