import numpy as np
import h5py
from random import shuffle
from IPython.display import clear_output
from skimage import io
from IPython.core.debugger import set_trace

def generate_hdf5(dataset):
    '''
    Generates a chunked hdf5 file in the video directory (dataset.directory) containing all frames in dataset.samples

    The generated file contains two datasets labeled 'X' and 'Y', both of which have the shape
    [dataset.length, dataset.sequence_length, 3, 256, 256]

    Access pattern of the data during training is non-consecutive random reads of the 0th dimension,
    and chunk size and cache size are set accordingly. 
    
    Parameters
    ----------
    dataset: video_data_classes.VideoDataset
        The dataset of video to create an hdf5 file from.
    '''
    with h5py.File(r'E:\hdf5_file.hdf5', 'w', rdcc_nbytes=(256 * 256 * 3 * dataset.sequence_length + 100000)) as f:
        # Default gzip compression is a balance between time to compress/uncompress and disk space. uint8 is used because this is image data. 
        X_data = f.create_dataset('X', (dataset.length, dataset.sequence_length, 3, 64, 64), chunks=(1, dataset.sequence_length, 3, 64, 64), compression='gzip', dtype='uint8')
        Y_data = f.create_dataset('Y', (dataset.length, dataset.sequence_length, 3, 256, 256), chunks=(1, dataset.sequence_length, 3, 256, 256), compression='gzip', dtype='uint8')

        count = 0
        for i, sample in list(enumerate(dataset.samples)):
            clear_output(wait=True)
            print(count)
            count += 1
            for j, frame in list(enumerate(sample)):
                X_data[i, j, :, :, :] = np.transpose(io.imread(frame[1]), (2, 0, 1)) # tranpose from sk-image style (h, w, c) to PyTorch style (c, h, w)
                Y_data[i, j, :, :, :] = np.transpose(io.imread(frame[0]), (2, 0, 1))
             
            
def make_batches(num_samples, batch_size):
    '''
    Creates batches of indices for training
    
    Parameters:
    -----------
    num_samples: int
        The number of samples to organize into batches
    batch_size: int
        The size of a batch.
        
    Returns
    -------
    np.ndarray
        A numpy array with each row as a batch. Has dimensions:
        [num_samples / batch_size, batch_size].
        Will automatically truncate list of sample indices to make num_samples divisible by batch_size.
    '''
    indices = list(range(num_samples))
    shuffle(indices)
    indices = indices[:num_samples - (num_samples % batch_size)]
    batch_indices = np.reshape(np.array(indices), (-1, batch_size))
    batch_indices = np.apply_along_axis(lambda x: np.sort(x), 1, batch_indices)
    return batch_indices