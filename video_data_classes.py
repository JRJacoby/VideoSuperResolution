"""This module contains classes for loading and organizing video data for input into the SuperResolutionNetwork
class."""

import os
import torch

from skimage import io
from torch.utils.data import Dataset
from IPython.core.debugger import set_trace

class VideoDataset(Dataset):
    """
    A representation of a list of video samples, each sample consisting of several frames.
    
    ...
    
    Attributes
    ----------
    samples : list
        A list of lists. Each inner list contains sequence_length number of strings. Each string is an absolute
        filepath to a frame of a video. Note that while only strings are stored, t
    length : int
        The number of video samples contained in the dataset
    sequence_length : int
        The number of frames in each video sample
    directory : str
        The absolute path to the directory containing all video folders. 
    transform : any of torchvision.transforms from the PyTorch package
        The transformations to apply to each video frame before returning the frame when called with 
        VideoDataset[sample_id]
    """
    def __init__(self, directory, sequence_length, transform=None):
        """
        Parameters
        ----------
        directory : str
            The directory containing the video data. The directory should be arranged like the output of the
            "Prepare Videos.ipynb" script. See that script for more details.
        sequence_length : int
            The number of frames in each video sample.
        transform : any of torchvision.transforms from the PyTorch package
            The transformations to apply to each video frame before returning the frame when called with 
            VideoDataset[sample_id]
        """
        samples = []
        video_folders = os.listdir(directory)
        
        for video_folder in video_folders:
            patch_folders = os.listdir(directory + '\\' + video_folder)
            patch_folders = [patch_folder for patch_folder in patch_folders if '.' not in patch_folder]
            
            for patch_folder in patch_folders:
                patch_folder_path = directory + '\\' + video_folder + '\\' + patch_folder + '\\'
                files = os.listdir(patch_folder_path)
                frames = list(filter(lambda x: x.endswith('.png'), files))
                frames = sorted(frames, key=lambda string: int(''.join(char for char in string if char.isdigit())))
                frames = [patch_folder_path + frame for frame in frames]
                
                num_frames = len(frames)
                for i in list(range(0, num_frames, sequence_length))[:-1]:
                    samples.append(frames[i : i + sequence_length])
        
        self.samples = samples
        self.length = len(self.samples)
        self.sequence_length = sequence_length
        self.directory = directory
        self.transform = transform
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, id):
        images = [io.imread(image) for image in self.samples[id]]
        if self.transform:
            images = self.transform(images)
        return images
    
class LightenImageTransformer(object):
    def __call__(self, sample):
        return {'X':sample, 'Y':[(frame + 30).clip(0, 255) for frame in sample]}

class MinMaxTransformer(object):
    def __call__(self, sample):
        normalized_x = []
        normalized_y = []
        for frame_x, frame_y in zip(sample['X'], sample['Y']):
            normalized_x.append(frame_x / 255)
            normalized_y.append(frame_y / 255)
            
        return {'X':normalized_x, 'Y':normalized_y}
                                  