# Name(s): Justice Mason, Arthur Yang
# Date: 02/21/2024

import os
import glob
import numpy as np
import warnings
import argparse
import pickle
from tqdm.contrib.concurrent import process_map  # or thread_map
from PIL import Image
from multiprocessing import Pool, cpu_count

from torchvision import  transforms


class PreProcess:
    '''
    This class takes rendered images and preprocesses them before dumping out a numpy array of all the data.

    '''
    def __init__(self, data_dir, traj_len, img_size=28, n_proc=1):
        self.data_dir = data_dir #os.path.join(data_dir, 'images')
        self.traj_len = traj_len
        self.img_size = img_size
        self.n_proc = n_proc
        self.transform = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor()])

    def traj_to_numpy(self, traj_dir):
        traj_images = glob.glob(os.path.join(traj_dir, '*.png'))
        traj_images.sort()

        # print(f'\n Im inside process_map wow: {len(traj_images)} \n')
        if len(traj_images) != self.traj_len:
            warn_msg = f'Some trajectories did not have the correct trajecory length ({self.traj_len})'
            warn_msg += f'\n\t directory: {traj_dir} \t length: {len(traj_images)}'

        data = np.array([self.transform(Image.open(fname)).numpy() for fname in traj_images])
        return data

    def datasets_to_numpy(self):
        # gather list of direcotries of image trajectories
        image_dirs = glob.glob(os.path.join(self.data_dir, '*'))
        image_dirs.sort()

        print(f'\n {len(image_dirs)} image directories found ... \n')

        all_trajectories = process_map(self.traj_to_numpy, image_dirs,
                                       max_workers=self.n_proc)

        print('\n Finished compiling trajectories ... \n')
        return np.stack(all_trajectories)

    def ignore_bad_traj(self, trajectories, image_dirs):
        '''Helper method to prune out bad trajectories.'''

        print('\n Evaluating trajectory quality... \n')

        lens = np.array([len(traj) for traj in trajectories])
        bad_trajs = lens != self.traj_len

        if np.sum(bad_trajs) > 0:
            warn_msg = f'Some trajectories did not have the correct trajecory length ({self.traj_len}). The following trajectories have been ignored'

            for traj_dir, bad_len in zip(np.array(image_dirs)[bad_trajs], lens[bad_trajs]):
                warn_msg += f'\n\t directory: {traj_dir} \t length: {bad_len}'
            
            warnings.warn(warn_msg)
        return np.array(trajectories, dtype=object)[~bad_trajs]

    def process(self, save_path):
        res = self.datasets_to_numpy()
        print(f'\n Saving to {save_path} ... \n')

        np.save(save_path, res)

def get_args():
    '''
    Example Call:
    $ python data_preprocess.py --data_dir tmp/images --save_path tmp/data.npy \
                                --traj_len 20   --img_size 28 -N 2
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        help='parent directory where images are stored')

    parser.add_argument('--save_name',
                        type=str,
                        help='filepath to save numpy dump')

    parser.add_argument('--traj_len',
                        type=int,
                        help="expected length of trajectory")

    parser.add_argument('-N', '--n_proc',
                        type=int,
                        default=1,
                        help='number of parallel processes to run')

    parser.add_argument('--img_size',
                            type=int,
                            default=28,
                            help='size of image')

    args = parser.parse_args()
    return args

 
# GPT code
def load_images_from_trajectory(trajectory_path):
    trajectory_images = []
    for file in sorted(os.listdir(trajectory_path)):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(trajectory_path, file)
            img = Image.open(file_path).convert('RGB')
            img_array = np.transpose(np.array(img), axes=(2, 0, 1)) * (1./255)
            trajectory_images.append(img_array)
    return np.array(trajectory_images)

def load_images_from_directory(directory):
    with Pool(processes=cpu_count()) as pool:
        trajectories = pool.map(load_images_from_trajectory, [os.path.join(directory, traj) for traj in sorted(os.listdir(directory))])
        
    return np.array(trajectories)

if __name__=='__main__':
    args = get_args()
    pp = PreProcess(args.data_dir,
                    args.traj_len,
                    img_size=args.img_size,
                    n_proc=args.n_proc)

    res = pp.process(os.path.join(args.data_dir, args.save_name))
