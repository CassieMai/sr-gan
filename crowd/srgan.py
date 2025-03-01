"""
Code for the crowd application.
"""
import json
import os
import random
from collections import defaultdict

import PIL.Image
import imageio
import scipy.misc
import matplotlib
import numpy as np
import torch
from scipy.stats import norm
import torchvision
from torch.utils.data import DataLoader

from crowd import data
from crowd.data import ExtractPatchForPosition, CrowdExample, ImageSlidingWindowDataset, CrowdDataset
from crowd.models import DCGenerator, KnnDenseNetCat
from crowd.shanghai_tech_data import ShanghaiTechFullImageDataset, ShanghaiTechTransformedDataset
from crowd.ucf_qnrf_data import UcfQnrfFullImageDataset, UcfQnrfTransformedDataset
from crowd.world_expo_data import WorldExpoFullImageDataset, WorldExpoTransformedDataset
from crowd.clustered_orange_data import ClusteredOrangeFullImageDataset, ClusteredOrangeTransformedDataset
from srgan import Experiment
from utility import MixtureModel, gpu
import PIL.Image as Image
import time

class CrowdExperiment(Experiment):
    """The crowd application."""

    def dataset_setup(self):
        """Sets up the datasets for the application."""
        settings = self.settings
        if settings.crowd_dataset == CrowdDataset.ucf_qnrf:
            self.dataset_class = UcfQnrfFullImageDataset
            self.train_dataset = UcfQnrfTransformedDataset(middle_transform=data.RandomHorizontalFlip(),
                                                           seed=settings.labeled_dataset_seed,
                                                           number_of_examples=settings.labeled_dataset_size,
                                                           map_directory_name=settings.map_directory_name)
            self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size,
                                                   pin_memory=self.settings.pin_memory,
                                                   num_workers=settings.number_of_data_workers)
            self.unlabeled_dataset = UcfQnrfTransformedDataset(middle_transform=data.RandomHorizontalFlip(),
                                                               seed=settings.labeled_dataset_seed,
                                                               number_of_examples=settings.unlabeled_dataset_size,
                                                               map_directory_name=settings.map_directory_name,
                                                               examples_start=settings.labeled_dataset_size)
            self.unlabeled_dataset_loader = DataLoader(self.unlabeled_dataset, batch_size=settings.batch_size,
                                                       pin_memory=self.settings.pin_memory,
                                                       num_workers=settings.number_of_data_workers)
            self.validation_dataset = UcfQnrfTransformedDataset(dataset='test', seed=101,
                                                                map_directory_name=settings.map_directory_name)
        elif settings.crowd_dataset == CrowdDataset.shanghai_tech:
            self.dataset_class = ShanghaiTechFullImageDataset
            self.train_dataset = ShanghaiTechTransformedDataset(middle_transform=data.RandomHorizontalFlip(),
                                                                seed=settings.labeled_dataset_seed,
                                                                number_of_examples=settings.labeled_dataset_size,
                                                                map_directory_name=settings.map_directory_name)
            self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size,
                                                   pin_memory=self.settings.pin_memory,
                                                   num_workers=settings.number_of_data_workers)
            self.unlabeled_dataset = ShanghaiTechTransformedDataset(middle_transform=data.RandomHorizontalFlip(),
                                                                    seed=100,
                                                                    number_of_examples=settings.unlabeled_dataset_size,
                                                                    map_directory_name=settings.map_directory_name)
            self.unlabeled_dataset_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size,
                                                       pin_memory=self.settings.pin_memory,
                                                       num_workers=settings.number_of_data_workers)
            self.validation_dataset = ShanghaiTechTransformedDataset(dataset='test', seed=101,
                                                                     map_directory_name=settings.map_directory_name)

        elif settings.crowd_dataset == CrowdDataset.world_expo:
            self.dataset_class = WorldExpoFullImageDataset
            self.train_dataset = WorldExpoTransformedDataset(middle_transform=data.RandomHorizontalFlip(),
                                                             seed=settings.labeled_dataset_seed,
                                                             number_of_cameras=settings.number_of_cameras,
                                                             number_of_images_per_camera=settings.number_of_images_per_camera)
            self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size,
                                                   pin_memory=self.settings.pin_memory,
                                                   num_workers=settings.number_of_data_workers)
            self.unlabeled_dataset = WorldExpoTransformedDataset(middle_transform=data.RandomHorizontalFlip(),
                                                                 seed=settings.labeled_dataset_seed,
                                                                 number_of_cameras=settings.number_of_cameras,
                                                                 number_of_images_per_camera=settings.number_of_images_per_camera)
            self.unlabeled_dataset_loader = DataLoader(self.unlabeled_dataset, batch_size=settings.batch_size,
                                                       pin_memory=self.settings.pin_memory,
                                                       num_workers=settings.number_of_data_workers)
            self.validation_dataset = WorldExpoTransformedDataset(dataset='validation', seed=101)
            if self.settings.batch_size > self.train_dataset.length:
                self.settings.batch_size = self.train_dataset.length

        elif settings.crowd_dataset == CrowdDataset.clustered_orange:
            print('crowd dataset: clustered_orange')
            self.dataset_class = ClusteredOrangeFullImageDataset
            self.train_dataset = ClusteredOrangeTransformedDataset(middle_transform=data.RandomHorizontalFlip(), seed=settings.labeled_dataset_seed, number_of_examples=settings.labeled_dataset_size, map_directory_name=settings.map_directory_name)
            # self.train_dataset = ClusteredOrangeFullImageDataset(dataset='train', seed=settings.labeled_dataset_seed, number_of_examples=settings.labeled_dataset_size, map_directory_name=settings.map_directory_name)
            self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size,
                                                   pin_memory=self.settings.pin_memory,
                                                   num_workers=settings.number_of_data_workers)
            self.unlabeled_dataset = ClusteredOrangeTransformedDataset(middle_transform=data.RandomHorizontalFlip(), seed=settings.labeled_dataset_seed, number_of_examples=settings.unlabeled_dataset_size, map_directory_name=settings.map_directory_name, examples_start=settings.labeled_dataset_size)
            # self.unlabeled_dataset = ClusteredOrangeFullImageDataset(dataset='train', seed=settings.labeled_dataset_seed, number_of_examples=settings.unlabeled_dataset_size, map_directory_name=settings.map_directory_name, examples_start=settings.labeled_dataset_size)
            self.unlabeled_dataset_loader = DataLoader(self.unlabeled_dataset, batch_size=settings.batch_size,
                                                       pin_memory=self.settings.pin_memory,
                                                       num_workers=settings.number_of_data_workers)
            self.validation_dataset = ClusteredOrangeTransformedDataset(dataset='test', seed=101,
                                                                map_directory_name=settings.map_directory_name)
            # self.validation_dataset = ClusteredOrangeFullImageDataset(dataset='test', seed=101, map_directory_name=settings.map_directory_name)

    def model_setup(self):
        """Prepares all the model architectures required for the application."""
        self.G = DCGenerator()
        self.D = KnnDenseNetCat()
        self.DNN = KnnDenseNetCat()

    def validation_summaries(self, step):
        """Prepares the summaries that should be run for the given application."""
        settings = self.settings
        dnn_summary_writer = self.dnn_summary_writer
        gan_summary_writer = self.gan_summary_writer
        DNN = self.DNN
        D = self.D
        G = self.G
        train_dataset = self.train_dataset
        validation_dataset = self.validation_dataset

        # DNN training evaluation.
        self.evaluation_epoch(settings, DNN, train_dataset, dnn_summary_writer, '2 Train Error', shuffle=False)
        # DNN validation evaluation.
        dnn_validation_count_mae = self.evaluation_epoch(settings, DNN, validation_dataset, dnn_summary_writer,
                                                         '1 Validation Error', shuffle=False)
        # GAN training evaluation.
        self.evaluation_epoch(settings, D, train_dataset, gan_summary_writer, '2 Train Error', shuffle=False)
        # GAN validation evaluation.
        self.evaluation_epoch(settings, D, validation_dataset, gan_summary_writer, '1 Validation Error',
                              comparison_value=dnn_validation_count_mae, shuffle=False)

        # Real images.
        train_iterator = iter(DataLoader(train_dataset, batch_size=settings.batch_size))
        images, densities, maps = next(train_iterator)
        predicted_densities, _, predicted_maps = D(images.to(gpu))
        real_comparison_image = self.create_map_comparison_image(images, maps, predicted_maps.to('cpu'))
        gan_summary_writer.add_image('Real', real_comparison_image)
        dnn_predicted_densities, _, predicted_maps = DNN(images.to(gpu))
        dnn_real_comparison_image = self.create_map_comparison_image(images, maps, predicted_maps.to('cpu'))
        dnn_summary_writer.add_image('Real', dnn_real_comparison_image)
        validation_iterator = iter(DataLoader(train_dataset, batch_size=settings.batch_size))
        images, densities, maps = next(validation_iterator)
        predicted_densities, _, predicted_maps = D(images.to(gpu))
        validation_comparison_image = self.create_map_comparison_image(images, maps, predicted_maps.to('cpu'))
        gan_summary_writer.add_image('Validation', validation_comparison_image)
        dnn_predicted_densities, _, predicted_maps = DNN(images.to(gpu))
        dnn_validation_comparison_image = self.create_map_comparison_image(images, maps, predicted_maps.to('cpu'))
        dnn_summary_writer.add_image('Validation', dnn_validation_comparison_image)

        # Generated images.
        z = torch.randn(settings.batch_size, G.input_size)
        fake_examples = G(z.to(gpu)).to('cpu')
        fake_images_image = torchvision.utils.make_grid(fake_examples.data[:9], normalize=True, range=(-1, 1), nrow=3)
        gan_summary_writer.add_image('Fake/Standard', fake_images_image.numpy())
        z = torch.as_tensor(MixtureModel([norm(-settings.mean_offset, 1), norm(settings.mean_offset, 1)]
                                         ).rvs(size=[settings.batch_size, G.input_size]).astype(np.float32))
        fake_examples = G(z.to(gpu)).to('cpu')
        fake_images_image = torchvision.utils.make_grid(fake_examples.data[:9], normalize=True, range=(-1, 1), nrow=3)
        gan_summary_writer.add_image('Fake/Offset', fake_images_image.numpy())

        self.test_summaries()


    def evaluation_epoch(self, settings, network, dataset, summary_writer, summary_name, comparison_value=None,
                         shuffle=True):
        """Runs the evaluation and summaries for the data in the dataset."""
        dataset_loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=shuffle)
        predicted_counts, densities, predicted_densities, maps, predicted_maps = np.array([]), np.array(
            []), np.array([]), np.array([]), np.array([])
        for index, (images, labels, batch_maps) in enumerate(dataset_loader):
            images, labels = images.to(gpu), labels.to(gpu)
            (batch_predicted_densities, batch_predicted_counts, batch_predicted_maps
             ) = self.images_to_predicted_labels(network, images)
            labels = labels.to('cpu')
            batch_predicted_densities = batch_predicted_densities.detach().to('cpu').numpy()
            batch_predicted_maps = batch_predicted_maps.detach().to('cpu').numpy()
            batch_predicted_counts = batch_predicted_counts.detach().to('cpu').numpy()
            predicted_counts = np.concatenate([predicted_counts, batch_predicted_counts])
            if predicted_densities.size == 0:
                predicted_densities = predicted_densities.reshape([0, *batch_predicted_densities.shape[1:]])
            predicted_densities = np.concatenate([predicted_densities, batch_predicted_densities])
            if densities.size == 0:
                densities = densities.reshape([0, *labels.shape[1:]])
            densities = np.concatenate([densities, labels])
            if maps.size == 0:
                maps = maps.reshape([0, *batch_maps.shape[1:]])
                maps = np.concatenate([maps, batch_maps])
            if predicted_maps.size == 0:
                predicted_maps = predicted_maps.reshape([0, *batch_predicted_maps.shape[1:]])
                predicted_maps = np.concatenate([predicted_maps, batch_predicted_maps])
            if index * self.settings.batch_size >= 100:
                break
        maps = np.expand_dims(maps, axis=1)
        count_me = (predicted_counts - densities.sum(1).sum(1)).mean()
        summary_writer.add_scalar('{}/ME'.format(summary_name), count_me)
        count_mae = np.abs(predicted_counts - densities.sum(1).sum(1)).mean()
        summary_writer.add_scalar('{}/MAE'.format(summary_name), count_mae)
        density_mae = np.abs(predicted_maps - maps).mean()
        summary_writer.add_scalar('{}/kNN MAE'.format(summary_name), density_mae)
        count_mse = (np.abs(predicted_counts - densities.sum(1).sum(1)) ** 2).mean()
        summary_writer.add_scalar('{}/MSE'.format(summary_name), count_mse)
        density_mse = (np.abs(predicted_maps - maps) ** 2).mean()
        summary_writer.add_scalar('{}/kNN MSE'.format(summary_name), density_mse)
        if comparison_value is not None:
            summary_writer.add_scalar('{}/Ratio MAE GAN DNN'.format(summary_name), count_mae / comparison_value)
        return count_mae

    def convert_density_maps_to_heatmaps(self, label, predicted_label):
        """
        Converts a label and predicted label density map into their respective heatmap images.

        :param label: The label tensor.
        :type label: torch.autograd.Variable
        :param predicted_label: The predicted labels tensor.
        :type predicted_label: torch.autograd.Variable
        :return: The heatmap label tensor and heatmap predicted label tensor.
        :rtype: (torch.autograd.Variable, torch.autograd.Variable)
        """
        mappable = matplotlib.cm.ScalarMappable(cmap='inferno')
        label_array = label.numpy()
        predicted_label_array = predicted_label.numpy()
        mappable.set_clim(vmin=min(label_array.min(), predicted_label_array.min()),
                          vmax=max(label_array.max(), predicted_label_array.max()))
        patch_size = self.settings.image_patch_size
        # resized_label_array = scipy.misc.imresize(label_array, (patch_size, patch_size), mode='F')
        resized_label_array = np.array(Image.fromarray(label_array).resize((patch_size, patch_size), PIL.Image.BICUBIC))  # xcmai
        label_heatmap_array = mappable.to_rgba(resized_label_array).astype(np.float32)
        label_heatmap_tensor = torch.as_tensor(label_heatmap_array[:, :, :3].transpose((2, 0, 1)))
        # resized_predicted_label_array = scipy.misc.imresize(predicted_label_array, (patch_size, patch_size), mode='F')
        resized_predicted_label_array = np.array(Image.fromarray(predicted_label_array).resize((patch_size, patch_size), PIL.Image.BICUBIC))  # xcmai
        predicted_label_heatmap_array = mappable.to_rgba(resized_predicted_label_array).astype(np.float32)
        predicted_label_heatmap_tensor = torch.as_tensor(predicted_label_heatmap_array[:, :, :3].transpose((2, 0, 1)))
        return label_heatmap_tensor, predicted_label_heatmap_tensor

    def create_map_comparison_image(self, images, labels, predicted_labels, number_of_images=3):
        """
        Creates a grid of images from the original images, the true maps, and each predicted map.

        :param images: The original RGB images.
        :type images: torch.autograd.Variable
        :param labels: The map labels.
        :type labels: torch.autograd.Variable
        :param predicted_labels: The predicted map labels.
        :type predicted_labels: torch.autograd.Variable
        :param number_of_images: The number of (original) images to include in the grid.
        :type number_of_images: int
        :return: The image of the grid of images.
        :rtype: np.ndarray
        """
        grid_image_list = []
        for image_index in range(min(number_of_images, images.size()[0])):
            grid_image_list.append((images[image_index].data + 1) / 2)
            for predicted_map_index in range(predicted_labels.size()[1]):
                label_heatmap, predicted_label_heatmap = self.convert_density_maps_to_heatmaps(
                    labels[image_index].data, predicted_labels[image_index, predicted_map_index].data
                )
                if predicted_map_index == 0:
                    grid_image_list.append(label_heatmap)
                grid_image_list.append(predicted_label_heatmap)
        return torchvision.utils.make_grid(grid_image_list, nrow=2 + predicted_labels.size()[1], normalize=True,
                                           range=(0, 1))

    def labeled_loss_function(self, predicted_labels, labels, order=2):
        """The loss function for the crowd application."""
        head_labels, map_labels = labels
        maps = map_labels.unsqueeze(1)
        predicted_density_labels, predicted_count_labels, predicted_maps = predicted_labels
        map_loss = (torch.abs(predicted_maps - maps)).mean(1).sum(1).sum(1).pow(order).mean()
        count_loss = torch.abs(predicted_count_labels - head_labels.sum(1).sum(1)).pow(order).mean()
        # count_loss = torch.abs(predicted_count_labels - map_labels.sum(1).sum(1)).pow(order).mean()  # xcmai
        return count_loss + (map_loss * self.settings.map_multiplier)

    def images_to_predicted_labels(self, network, images):
        """Runs the code to go from images to a predicted labels. Useful for overriding."""
        predicted_densities, predicted_counts, predicted_maps = network(images)
        return predicted_densities, predicted_counts, predicted_maps

    def test_summaries(self):
        """Evaluates the model on test data during training."""
        print('in test_summaries')
        test_dataset = self.dataset_class(dataset='test', map_directory_name=self.settings.map_directory_name)
        if self.settings.test_summary_size is not None:
            indexes = random.sample(range(test_dataset.length), self.settings.test_summary_size)
        else:
            indexes = range(test_dataset.length)
        dnn_mae_count = None
        dnn_rmse_count = None
        for network in [self.DNN, self.D]:
            totals = defaultdict(lambda: 0)
            for index in indexes:
                print('index', index)
                full_image, full_label, full_maps = test_dataset[index]
                full_example = CrowdExample(image=full_image, label=full_label)
                sss = time.time()
                full_predicted_count, full_predicted_label = self.predict_full_example(full_example, network)
                print('predict_full_example running time ', time.time() - sss)
                totals['Count error'] += np.abs(full_predicted_count - full_example.label.sum())
                totals['NAE'] += np.abs(full_predicted_count - full_example.label.sum()) / (full_example.label.sum() + 1e-10)
                totals['Density sum error'] += np.abs(full_predicted_label.sum() - full_example.label.sum())
                totals['SE count'] += (full_predicted_count - full_example.label.sum()) ** 2
                totals['SE density'] += (full_predicted_label.sum() - full_example.label.sum()) ** 2
                totals['Pred. count sum'] += full_predicted_count  # xcmai
                totals['GT count sum'] += full_example.label.sum()  # xcmai
            if network is self.DNN:
                summary_writer = self.dnn_summary_writer
            else:
                summary_writer = self.gan_summary_writer
            nae_count = totals['NAE'] / len(indexes)
            summary_writer.add_scalar('0 Test Error/NAE count', nae_count)
            mae_count = totals['Count error'] / len(indexes)
            summary_writer.add_scalar('0 Test Error/MAE count', mae_count)
            mae_density = totals['Density sum error'] / len(indexes)
            summary_writer.add_scalar('0 Test Error/MAE density', mae_density)
            rmse_count = (totals['SE count'] / len(indexes)) ** 0.5
            summary_writer.add_scalar('0 Test Error/RMSE count', rmse_count)
            rmse_density = (totals['SE density'] / len(indexes)) ** 0.5
            summary_writer.add_scalar('0 Test Error/RMSE density', rmse_density)
            rc_density = (totals['Pred. count sum'] / totals['GT count sum'])  # xcmai
            summary_writer.add_scalar('0 Test Error/Ratio of counting', rc_density) # xcmai
            print('len indexes', len(indexes))
            print('nae_count', nae_count)
            print('mae_count', mae_count)
            print('mae_density', mae_density)
            print('rmse_count', rmse_count)
            print('rmse_density', rmse_density)
            print('rc_density', rc_density)

            if network is self.DNN:
                dnn_mae_count = mae_count
                dnn_rmse_count = rmse_count
            else:
                summary_writer.add_scalar('0 Test Error/Ratio MAE GAN DNN', mae_count / dnn_mae_count)
                summary_writer.add_scalar('0 Test Error/Ratio RMSE GAN DNN', rmse_count / dnn_rmse_count)
            print('totals', totals)
        print('test_summaries end')

        # visualization. xcmai
        from utility import visualize2
        visualize2(full_image, full_label, full_predicted_label)

    def evaluate(self, during_training=False, step=None, number_of_examples=None):
        """Evaluates the model on test data."""
        super().evaluate()
        for network in [self.DNN, self.D]:
            # test_dataset = self.settings.dataset_class(dataset='test')
            test_dataset = self.dataset_class(dataset='test')
            totals = defaultdict(lambda: 0)
            print('test_dataset', len(test_dataset))
            for full_example_index, (full_image, full_label) in enumerate(test_dataset):
                print('Processing full example {}...'.format(full_example_index), end='\r')
                full_example = CrowdExample(image=full_image, label=full_label)
                full_predicted_count, full_predicted_label = self.predict_full_example(full_example, network)
                totals['Count'] += full_example.label.sum()
                totals['Density error'] += np.abs(full_predicted_label - full_example.label).sum()
                totals['Count error'] += np.abs(full_predicted_count - full_example.label.sum())
                totals['Density sum error'] += np.abs(full_predicted_label.sum() - full_example.label.sum())
                totals['Predicted count'] += full_predicted_count
                totals['Predicted density sum'] += full_predicted_label.sum()
                totals['SE count'] += (full_predicted_count - full_example.label.sum()) ** 2
                totals['SE density'] += (full_predicted_label.sum() - full_example.label.sum()) ** 2
            else:
                if network is self.DNN:
                    print('=== DNN ===')
                else:
                    print('=== GAN ===')
                print('MAE count: {}'.format(totals['Count error'] / len(test_dataset)))
                print('MAE density: {}'.format(totals['Density sum error'] / len(test_dataset)))
                print('MSE count: {}'.format(totals['SE count'] / len(test_dataset)))
                print('MSE density: {}'.format(totals['SE density'] / len(test_dataset)))
                for key, value in totals.items():
                    print('Total {}: {}'.format(key, value))

    def predict_full_example(self, full_example, network):
        """
        Runs the prediction for a full example, by processing patches and averaging the patch results.

        :param full_example: The full crowd example to be processed.
        :type full_example: CrowdExample
        :param network: The network to process the patches.
        :type network: torch.nn.Module
        :return: The predicted count array and density array.
        :rtype: (np.ndarray, np.ndarray)
        """
        sum_density_label = np.zeros_like(full_example.label, dtype=np.float32)
        sum_count_label = np.zeros_like(full_example.label, dtype=np.float32)
        hit_predicted_label = np.zeros_like(full_example.label, dtype=np.int32)
        full_example_dataset = ImageSlidingWindowDataset(full_example, self.settings.image_patch_size,
                                                         self.settings.test_sliding_window_size)
        full_example_dataloader = DataLoader(full_example_dataset, batch_size=self.settings.batch_size,
                                             pin_memory=self.settings.pin_memory,
                                             num_workers=self.settings.number_of_data_workers)
        patch_size = self.settings.image_patch_size
        for batch in full_example_dataloader:
            images = torch.stack([image for image in batch[0]])
            predicted_labels, predicted_counts, predicted_maps = network(images.to(gpu))
            predicted_labels, predicted_counts = predicted_labels.to('cpu'), predicted_counts.to('cpu')
            for example_index, image in enumerate(batch[0]):
                x = batch[1][example_index]
                y = batch[2][example_index]
                predicted_label = predicted_labels[example_index].detach().numpy()
                try:
                    predicted_count = predicted_counts[example_index].detach().numpy()
                except IndexError:
                    predicted_count = predicted_counts.detach().numpy()
                # predicted_label = scipy.misc.imresize(predicted_label, (patch_size, patch_size), mode='F')
                predicted_label = np.array(Image.fromarray(predicted_label).resize((patch_size, patch_size), PIL.Image.BICUBIC))  # xcmai

                predicted_label = predicted_label
                predicted_count_array = np.full(predicted_label.shape,
                                                predicted_count / predicted_label.size)
                half_patch_size = patch_size // 2
                y_start_offset = 0
                if y - half_patch_size < 0:
                    y_start_offset = half_patch_size - y
                y_end_offset = 0
                if y + half_patch_size > full_example.label.shape[0]:
                    y_end_offset = y + half_patch_size - full_example.label.shape[0]
                x_start_offset = 0
                if x - half_patch_size < 0:
                    x_start_offset = half_patch_size - x
                x_end_offset = 0
                if x + half_patch_size > full_example.label.shape[1]:
                    x_end_offset = x + half_patch_size - full_example.label.shape[1]
                sum_density_label[y - half_patch_size + y_start_offset:y + half_patch_size - y_end_offset,
                                  x - half_patch_size + x_start_offset:x + half_patch_size - x_end_offset
                                  ] += predicted_label[y_start_offset:predicted_label.shape[0] - y_end_offset,
                                                       x_start_offset:predicted_label.shape[1] - x_end_offset]
                sum_count_label[y - half_patch_size + y_start_offset:y + half_patch_size - y_end_offset,
                                x - half_patch_size + x_start_offset:x + half_patch_size - x_end_offset
                                ] += predicted_count_array[y_start_offset:predicted_count_array.shape[0] - y_end_offset,
                                                           x_start_offset:predicted_count_array.shape[1] - x_end_offset]
                hit_predicted_label[y - half_patch_size + y_start_offset:y + half_patch_size - y_end_offset,
                                    x - half_patch_size + x_start_offset:x + half_patch_size - x_end_offset
                                    ] += 1
        hit_predicted_label[hit_predicted_label == 0] = 1
        full_predicted_label = sum_density_label / hit_predicted_label.astype(np.float32)
        full_predicted_count = np.sum(sum_count_label / hit_predicted_label.astype(np.float32))
        return full_predicted_count, full_predicted_label

    def batches_of_patches_with_position(self, full_example, window_step_size=32):
        """
        A generator for extracting patches from an image in batches.

        :param full_example: The full example to be patched.
        :type full_example: CrowdExample
        :param window_step_size: The sliding window size.
        :type window_step_size: int
        :return: A batch of patches.
        :rtype: list[list[CrowdExample]]
        """
        extract_patch_transform = ExtractPatchForPosition()
        test_transform = torchvision.transforms.Compose([data.NegativeOneToOneNormalizeImage(),
                                                         data.NumpyArraysToTorchTensors()])
        batch = []
        for y in range(0, full_example.label.shape[0], window_step_size):
            for x in range(0, full_example.label.shape[1], window_step_size):
                patch = extract_patch_transform(full_example, y, x)
                example = test_transform(patch)
                example_with_position = CrowdExample(image=example.image, label=example.label,
                                                     patch_center_x=x, patch_center_y=y)
                batch.append(example_with_position)
                if len(batch) == self.settings.batch_size:
                    yield batch
                    batch = []
        yield batch

    def inference(self, input_):
        """
        Run the inference for crowd detection.
        """
        label = np.zeros(input_.shape[:2], dtype=np.float32)
        example = CrowdExample(image=input_, label=label)
        import datetime
        start = datetime.datetime.now()
        with torch.no_grad():
            predicted_count, predicted_label = self.predict_full_example(full_example=example,
                                                                         network=self.inference_network)
        print(datetime.datetime.now() - start)
        return predicted_count, predicted_label
