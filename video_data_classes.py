import os

from skimage import io
from torch.utils.data import Dataset
from IPython.core.debugger import set_trace

class VideoDataset(Dataset):
    def __init__(self, directory, sequence_length, transform=None):
        samples = []
        video_folders = os.listdir(directory)
        
        for folder in video_folders:
            files = os.listdir(directory + '\\' + folder)
            frames = list(filter(lambda x: x.endswith('.png'), files))
            frames = [directory + '\\' + folder + '\\' + frame for frame in frames]
            frames = sorted(frames, key=lambda string: int(''.join(char for char in string if char.isdigit())))
            
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
                                  