import numpy as np
import torch

def rot_x(theta: np.ndarray):
    """"""
    t = np.radians(theta)
    ct = np.cos(t)
    st = np.sin(t)

    DCM = np.array([[1. , 0., 0.],
                    [0., ct, st],
                    [0., -st, ct]])
    return DCM

def rot_y(theta: np.ndarray):
    """"""
    t = np.radians(theta)
    ct = np.cos(t)
    st = np.sin(t)

    DCM = np.array([[ct , 0., -st],
                    [0., 1., 0.],
                    [st, 0., ct]])
    return DCM

def rot_z(theta: np.ndarray):
    """"""
    t = np.radians(theta)
    ct = np.cos(t)
    st = np.sin(t)

    DCM = np.array([[ct , st, 0.],
                    [-st, ct, 0.],
                    [0., 0., 1.]])
    return DCM

def rot_x_batch(angle_arr):
    ''''''
    output = []

    for i in range(angle_arr.shape[0]):
        angle = angle_arr[i, ...]
        R_ = rot_x(angle)
        output.append(R_)

    Rx_batch = np.stack(output, axis=0)
    return Rx_batch

def rot_y_batch(angle_arr):
    ''''''
    output = []

    for i in range(angle_arr.shape[0]):
        angle = angle_arr[i, ...]
        R_ = rot_y(angle)
        output.append(R_)

    Ry_batch = np.stack(output, axis=0)
    return Ry_batch

def rot_z_batch(angle_arr):
    ''''''
    output = []

    for i in range(angle_arr.shape[0]):
        angle = angle_arr[i, ...]
        R_ = rot_z(angle)
        output.append(R_)

    Rz_batch = np.stack(output, axis=0)
    return Rz_batch

def eazyz_to_group_matrix(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """"""
    phi = alpha
    theta = beta
    psi = gamma

    B = rot_z_batch(phi)
    C = rot_y_batch(theta)
    D = rot_z_batch(psi)

    A = np.einsum('bij, bjk, bkl -> bil', B, C, D, optimize='optimal')
    return A

def eazxz_to_group_matrix(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """"""
    phi = alpha
    theta = beta
    psi = gamma

    D = rot_z_batch(phi)
    C = rot_x_batch(theta)
    B = rot_z_batch(psi)

    A = np.einsum('bij,bjk,bkl->bil', B, C, D, optimize='optimal')
    return A

def pd_matrix(diag: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
    """
    Function constructing postive-definite matrix from diag/off-diag entries.

    ...

    Parameters
    ----------
    diag : torch.Tensor
        Diagonal elements of PD matrix.
        
    off-diag: torch.Tensor
        Off-diagonal elements of PD matrix.
        
    Returns
    -------
    matrix_pd : torch.Tensor
        Calculated PD matrix.
        
    Notes
    -----

    """
    diag_dim = diag.shape[0]

    L = torch.diag_embed(diag)
    ind = np.tril_indices(diag_dim, k=-1)
    flat_ind  = np.ravel_multi_index(ind, (diag_dim, diag_dim))

    L = torch.flatten(L, start_dim=0)
    L[flat_ind] = off_diag
    L = torch.reshape(L, (diag_dim, diag_dim))

    matrix_pd = L @ L.T + (0.001 * torch.eye(3, device=diag.device))

    return matrix_pd

def nonuniform_moi_calc(l: float, w: float, h: float, f: float, mass: float = 1.):
    moi_cm = (1/12.) * (1 - f) * mass * np.diag([w**2 + h**2, l**2 + h**2, l**2 + w**2])
    r_c = np.array([-l/2, -w/2, -h/2])
    moi = moi_cm + (f * mass * (np.dot(r_c, r_c) * np.eye(3) - np.outer(r_c, r_c)))

    return moi