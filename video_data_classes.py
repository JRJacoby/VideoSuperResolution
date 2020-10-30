"""This module contains classes for loading and organizing video data for input into the SuperResolutionNetwork
class."""

import os
import torch
import h5py
import numpy as np

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
    def __init__(self, directory, sequence_length):
        """
        Parameters
        ----------
        directory : str
            The directory containing the video data. The directory should be arranged like the output of the
            "Prepare Videos.ipynb" script. See that script for more details.
        sequence_length : int
            The number of frames in each video sample.
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
                LR_frames = [x.replace(r'Raw, Half-Size, and PNGs', r'LowResFrames') for x in frames]
                
                num_frames = len(frames)
                for i in list(range(0, num_frames, sequence_length))[:-1]:
                    samples.append(list(zip(frames[i : i + sequence_length], LR_frames[i : i + sequence_length])))
        
        self.samples = samples
        self.length = len(self.samples)
        self.sequence_length = sequence_length
        self.directory = directory
        
    def __len__(self):
        '''
        Number of training samples in dataset.
        
        Returns
        -------
        int
            Number of training samples in dataset
        '''
        return self.length
    
    def __getitem__(self, id):
        '''
        Returns one sample of input video and the corresponding sample of output video. Labeled 'X' and 'Y', respectively.
        Each sample has dimensions [self.sequence_length, 3, 256, 256]
        
        Parameters
        ----------
        id: int
            The id of the video sample to return.
            
        Returns
        -------
        dict
            {'X': input_video_sample, 'Y':output_video_sample}
        '''
        num_channels = 3 * self.sequence_length
        return {'X':self.compressed_data['X'][:, :, id * num_channels:id * (num_channels + 1)] / 255,
                'Y':self.compressed_data['Y'][:, :, id * num_channels:id * (num_channels + 1)] / 255}
                   