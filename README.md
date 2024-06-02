# Image Datasets for 3D Rigid Body Dynamics on **SO(**3**)**
Repository for implmentation of code used for data generation in ["Learning to Predict 3D Rotational Dynamics from Images of a Rigid Body with Unknown Mass Distribution"](https://www.mdpi.com/2226-4310/10/11/921). 

In many real-world settings, image observations of freely rotating 3D rigid bodies may be available when low-dimensional measurements are not. However, the high-dimensionality of image data precludes the use of classical estimation techniques to learn the dynamics. The usefulness of standard deep learning methods is also limited, because an image of a rigid body reveals nothing about the distribution of mass inside the body, which, together with initial angular velocity, is what
determines how the body will rotate. We present a physics-based neural network model to estimate and predict 3D rotational dynamics from image sequences. We achieve this using a multi-stage prediction pipeline that maps individual images to a latent representation homeomorphic to SO(3), computes angular velocities from latent pairs, and predicts future latent states using the Hamiltonian equations of motion. **In order to demonstrate the efficacy of our approach, we've created a framework to generate datasets of sequences of synthetic images of rotating objects, including cubes, prisms and satellites, with unknown uniform and non-uniform mass distributions. This repository provides a way to modularly generate different datasets of 3D objects rotating and moving accoding to 3D rigid body dynamics.**

## Dependencies
The dependencies for this repository are given in the '''environment.yaml''' file. Some primary dependencies include:\

- [Pytorch](https://pytorch.org/) (torch => 2.0.1)
- [BlenderPy](https://pypi.org/project/bpy/) (bpy => 4.0.0)
- [Numpy](https://numpy.org/install/) (numpy => 1.24.3)

## Example Usage 
