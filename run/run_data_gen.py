# Name(s): Justice Mason
# Date: 02/21/24
import sys, os
import argparse
from datetime import datetime

import numpy as np
import torch, pytorch3d
import random 

from src.rbnn_data_generation.utils.data_gen import build_V_gravity, generate_lowdim_dataset_3DP, generate_lowdim_dataset_heavytop, generate_lowdim_dataset, LieGroupVaritationalIntegrator, LieGroupVaritationalIntegrator_FRB
from src.rbnn_data_generation.utils.data_gen import make_gif
from src.rbnn_data_generation.utils.math_utils import pd_matrix
from src.rbnn_data_generation.utils.data_utils import ImageGeneratorBatch
from src.rbnn_data_generation.utils.data_preprocess import PreProcess, load_images_from_directory

# Reproducibility
def setup_reproducibility(seed: int):
    """"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Get args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="dataset name",
        required=True,
    )
    parser.add_argument(
        "--experiment_type",
        choices=['ucube', 'ncube', 'uprism', 'nprism', 'calipso', 'cloudsat', '3dpend', 'heavytop'],
        help="set experiment filename",
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="directory where dataset will be save",
        required=True,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="directory where the Blender models are located",
        required=True,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Set number of samples in the dataset",
        required=True
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=1.0,
        help="total mass of object",
        required=False
    )
    parser.add_argument(
        "--l3",
        type=float,
        default=1.0,
        help="distance vector from pivot to CoM",
        required=False
    )
    parser.add_argument(
        "--rho",
        type=float,
        nargs='+',
        help="distance vector from pivot to CoM",
        required=False
    )
    parser.add_argument(
        "--traj_len",
        type=int,
        default=100,
        help="set trajectory length, default: 100",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1e-3,
        help="set dt, default: 1e-3",
    )
    parser.add_argument(
        "--moi_diag_gt",
        type=float,
        nargs='+',
        default=None,
        required=True,
        help="set ground-truth moi diagonal entries, default: None",
    )
    parser.add_argument(
        "--moi_off_diag_gt",
        type=float,
        nargs='+',
        default=None,
        required=True,
        help="set ground-truth moi off-diagonal entries, default: None",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=64,
        help="rendered image size",
        required=True,
    )
    parser.add_argument(
        "--img_ratio",
        type=int,
        default=1,
        help="rendered image ratio",
    )
    parser.add_argument(
        "--img_quality",
        type=int,
        default=90,
        help="rendered image quality",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=50.0,
        help="set angular momentum sphere radius, default: 50.0",
    )
    parser.add_argument(
        "--R_ic_type",
        choices=['stable', 'unstable', 'uniform'],
        default='stable',
        help="set R ic type for sampling group matrices on SO(3)",
    )
    parser.add_argument(
        "--pi_ic_type",
        choices=['random', 'unstable', 'desired'],
        default='random',
        help="set pi ic types for sample angular momentum sphere",
    )
    parser.add_argument('-N', '--n_proc',
                        type=int,
                        default=2,
                        help='number of parallel processes to run'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set random seed",
        required=True
    )
    
    args = parser.parse_args()
    return args

# Main function
def main(args):
    """ 
    Main function for generating datasets.
    

    Notes
    -----
    Examples call python3 run_data_gen.py 

    """
    # Generate trajectories based on dynamics type

    # Set up reproducibility
    setup_reproducibility(args.seed)
    datestr = datetime.today().strftime('%m%d%Y')

    # Choose integrator
    if args.experiment_type in ["3dpend", "heavytop"]:
        # Constant of motion
        g = 9.81 # [m/s2]
        e_3 = torch.tensor([[0., 0., 1.]], dtype=torch.float64).T
        rho_gt = torch.tensor(args.rho, dtype=torch.float64)

        # Initialize potential function
        V_gravity = lambda R: build_V_gravity(m=args.mass, g=g, e_3=e_3, R=R, rho_gt=rho_gt)

        print(f"\n Experiment type: {args.experiment_type.upper()} using LGVI with gravity ... \n") 
        integrator = LieGroupVaritationalIntegrator()

    else:
        print(f"\n Experiment type: {args.experiment_type.upper()} using LGVI without gravity ... \n") 
        integrator = LieGroupVaritationalIntegrator_FRB()

    # Calculate MOI
    moi = pd_matrix(diag=torch.sqrt(torch.tensor(args.moi_diag_gt)), off_diag=torch.tensor(args.moi_off_diag_gt))

    # 3D Pendulum
    if args.experiment_type == '3dpend':
        model_path = args.model_dir + '3dpend.blend'
        object_name = 'Cube'

        data_R, data_pi = generate_lowdim_dataset_3DP(MOI=moi, 
                                V=V_gravity,
                                radius=args.radius, 
                                n_samples=args.n_samples, 
                                integrator=integrator,
                                timestep=args.dt,
                                traj_len=args.traj_len,
                                R_ic_type=args.R_ic_type,
                                pi_ic_type=args.pi_ic_type,
                                seed=args.seed)

    # Heavytop
    elif args.experiment_type == 'heavytop':
        model_path = args.model_dir + 'heavytop.blend'
        object_name = 'Cone'

        data_R, data_pi = generate_lowdim_dataset_heavytop(args=args,
                                        MOI=moi,
                                        mass=args.mass,
                                        l3=args.l3, 
                                        n_samples=args.n_samples, 
                                        integrator=integrator, 
                                        general_flag=True, 
                                        timestep=args.dt, 
                                        traj_len=args.traj_len, 
                                        V=V_gravity, 
                                        seed=args.seed)

    # CALIPSO, CloudSAT, prism, and cube
    else:
        if args.experiment_type[1:] == 'cube':
            model_path = args.model_dir + 'cube.blend'
            object_name = 'Cube'

        elif args.experiment_type[1:] == 'prism':
            model_path = args.model_dir + 'prism.blend'
            object_name = 'Cube'

        else:
            model_path = args.model_dir + args.experiment_type + '.blend'

            if args.experiment_type == 'calipso':
                object_name = 'Calipso'
            elif args.experiment_type == 'cloudsat':
                object_name = 'CloudSat'

        data_R, data_pi = generate_lowdim_dataset(MOI=moi, 
                                        radius=args.radius, 
                                        n_samples=args.n_samples, 
                                        integrator=integrator, 
                                        timestep=args.dt, 
                                        traj_len=args.traj_len, 
                                        bandwidth_us=5., 
                                        desired_samples=None, 
                                        ic_type='random', 
                                        V=None, 
                                        seed=args.seed)

    # Generate Images
    save_dir = f'{args.save_dir}/{args.experiment_type}/{datestr}/{args.dataset_name}/'

    # Make data/save directory if it doesn't exits
    os.makedirs(save_dir, exist_ok=True)

    # Generate images
    image_generator = ImageGeneratorBatch(R=data_R, object=object_name, savepath=save_dir, filepath=model_path, size=args.img_size, ratio=args.img_ratio, quality=args.img_quality)
    image_generator.generate_image()

    # Convert images to numpy array
    print('\n Processing images in batches ... \n')
    # Load images into a NumPy array
    res = load_images_from_directory(save_dir)

    # Print the shape of the resulting array
    print("Shape of the image array:", res.shape)

    save_path = os.path.join(save_dir, 'data.npy')
    print(f'\n Saving to {save_path} ... \n')

    np.save(save_path, res)
    
    # Save the ground-truth R, pi arrays
    np.save(arr=data_R.cpu().detach().numpy(), file=os.path.join(save_dir, 'data_R.npy'), allow_pickle=True)
    np.save(arr=data_pi.cpu().detach().numpy(), file=os.path.join(save_dir, 'data_pi.npy'), allow_pickle=True)

    # Create GIF
    make_gif(frame_folder=save_dir+'traj-0', gif_name=args.experiment_type+'-0', save_dir=save_dir)
    make_gif(frame_folder=save_dir+'traj-1', gif_name=args.experiment_type+'-1', save_dir=save_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)

