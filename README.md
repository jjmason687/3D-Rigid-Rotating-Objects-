# Image Datasets for 3D Rigid Body Dynamics on **SO(**3**)**
Repository for implmentation of code used for data generation in ["Learning to Predict 3D Rotational Dynamics from Images of a Rigid Body with Unknown Mass Distribution"](https://www.mdpi.com/2226-4310/10/11/921). 

 In order to demonstrate the efficacy of our approach, we've created a framework to generate datasets of sequences of synthetic images of rotating objects, including cubes, prisms and satellites, with unknown uniform and non-uniform mass distributions. This repository provides a way to modularly generate different datasets of 3D objects rotating and moving accoding to 3D rigid body dynamics. We use Blender as our rendering engine and Python (```numpy``` and ```torch```) to compute our state trajectories.

## Dependencies
The dependencies for this repository are given in the ```environment.yaml``` file. Some primary dependencies include:

- [Pytorch](https://pytorch.org/) (torch => 2.0.1)
- [BlenderPy](https://pypi.org/project/bpy/) (bpy => 4.0.0)
- [Numpy](https://numpy.org/install/) (numpy => 1.24.3)

Our implementation uses [Blender]() (Blender 4.1.1) to render our images. BlenderPy seems to require a graphical backend to run. Pre-generated datasets of 32-by-32 pixels can be found at this [link](https://www.dropbox.com/scl/fo/8m2vs7vuskyge0ivabauf/ANsTmZZtmV57Qm6p1GKKyrI?rlkey=33tjibx07270t2my3iozll0jq&st=15rvlnbz&dl=0).

## Example Usage 
An example call is give in the bash file ```run/run.sh```:
```
python3 run/run_data_gen.py --dataset_name='example_name'\
                        --experiment_type='ucube'\
                        --save_dir='example_dir'\
                        --model_dir='BlenderModel/'\
                        --n_samples=2\
                        --radius=50.\
                        --traj_len=100\
                        --dt=0.001\
                        --moi_diag_gt 0.602 0.602 0.602\
                        --moi_off_diag_gt 0. 0. 0.\
                        --img_size=64\
                        --img_ratio=1\
                        --img_quality=90\
                        --seed=0\
```
