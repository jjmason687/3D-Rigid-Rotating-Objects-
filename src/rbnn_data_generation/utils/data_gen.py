import numpy as np
import torch 
from pytorch3d.transforms import rotation_conversions as rc 

import glob
from PIL import Image

# Justice's code
class LieGroupVaritationalIntegrator_FRB():
    """
    
    """
    def __init__(self):
        super(LieGroupVaritationalIntegrator_FRB, self).__init__()
        
    def skew(self, v: torch.Tensor):
        
        S = torch.zeros([v.shape[0], 3, 3], dtype=torch.float64, device=v.device)
        S[:, 0, 1] = -v[..., 2]
        S[:, 1, 0] = v[..., 2]
        S[:, 0, 2] = v[..., 1]
        S[:, 2, 0] = -v[..., 1]
        S[:, 1, 2] = -v[..., 0]
        S[:, 2, 1] = v[..., 0]
    
        return S
    
    def cayley_transx(self, fc: torch.Tensor):
        """
        """
        F = torch.einsum('bij, bjk -> bik', (torch.eye(3, dtype=torch.float64, device=fc.device) + self.skew(fc)), torch.linalg.inv(torch.eye(3, dtype=torch.float64, device=fc.device) - self.skew(fc)))
        return F
    
    def calc_fc_init(self, a_vec: torch.Tensor, moi:torch.Tensor) -> torch.Tensor:
        """
        """
        fc_init = torch.einsum('bij, bj -> bi', torch.linalg.inv(2 * moi - self.skew(a_vec)), a_vec)
        return fc_init
    
    def calc_Ac(self, a_vec: torch.Tensor, moi: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        """
        
        Ac = a_vec + torch.einsum('bij, bj -> bi', self.skew(a_vec), fc) + torch.einsum('bj, b -> bj', fc, torch.einsum('bj, bj -> b', a_vec, fc)) - (2 * torch.einsum('ij, bj -> bi', moi, fc))
        return Ac
        
    def calc_grad_Ac(self, a_vec: torch.Tensor, moi: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        """
        grad_Ac = self.skew(a_vec) + torch.einsum('b, bij -> bij', torch.einsum('bi, bi -> b', a_vec, fc), torch.unsqueeze(torch.eye(3, dtype=torch.float64, device=fc.device), 0).repeat(fc.shape[0], 1, 1)) + torch.einsum('bi, bj -> bij', fc, a_vec) - (2 * moi)
        return grad_Ac
    
    def optimize_fc(self, pi_vec: torch.Tensor, moi: torch.Tensor, fc_list: list = [], timestep: float = 1e-3, max_iter: int = 100, tol: float = 1e-8) -> list:
        """
        """
        it = 0
        
        if not fc_list:
            a_vec = timestep * pi_vec
            fc_list.append(self.calc_fc_init(a_vec=a_vec, moi=moi))
        
        eps = torch.ones(fc_list[-1].shape[0], dtype=torch.float64, device=pi_vec.device)
        
        while  torch.any(eps > tol) and it < max_iter:
            fc_i = fc_list[-1]
            a_vec = timestep * pi_vec
            
            Ac = self.calc_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
            grad_Ac = self.calc_grad_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
           
            fc_ii = fc_i - torch.einsum('bij, bj -> bi', torch.linalg.inv(grad_Ac),  Ac)
            
            eps = torch.linalg.norm(fc_ii - fc_i, axis=-1)
            fc_list.append(fc_ii)
            it += 1
            
        return fc_list
    
    def step(self, R_i: torch.Tensor, pi_i: torch.Tensor, moi: torch.Tensor, fc_list: list = [], timestep: float = 1e-3):
        """
        """
        fc_list = self.optimize_fc(pi_vec=pi_i, moi=moi, timestep=timestep, fc_list=fc_list)
        
        fc_opt = fc_list[-1]
        F_i = self.cayley_transx(fc=fc_opt)
        
        R_ii = torch.einsum('bij, bjk -> bik', R_i, F_i)
        pi_ii = torch.einsum('bji, bj -> bi', F_i, pi_i)
        
        return R_ii, pi_ii, fc_list
    
    def integrate(self, pi_init: torch.Tensor, R_init: torch.Tensor, moi: torch.Tensor, timestep: float = 1e-3, traj_len: int = 100):
        """
        """
        pi_list = [pi_init.double()]
        R_list = [R_init.double()]
        moi = moi.double()
        
        for it in range(1, traj_len):
            fc_list = []
            R_i = R_list[-1]
            pi_i = pi_list[-1]
            
            R_ii, pi_ii, fc_list = self.step(R_i=R_i, pi_i=pi_i, moi=moi, fc_list=fc_list, timestep=timestep)
            
            R_list.append(R_ii)
            pi_list.append(pi_ii)
        
        R_traj = torch.stack(R_list, axis=1)
        pi_traj = torch.stack(pi_list, axis=1)
        return R_traj, pi_traj

class LieGroupVaritationalIntegrator():
    """
    
    """
    def __init__(self):
        super().__init__()
        
    def skew(self, v: torch.Tensor):
        
        S = torch.zeros([v.shape[0], 3, 3], dtype=torch.float64)
        S[:, 0, 1] = -v[..., 2]
        S[:, 1, 0] = v[..., 2]
        S[:, 0, 2] = v[..., 1]
        S[:, 2, 0] = -v[..., 1]
        S[:, 1, 2] = -v[..., 0]
        S[:, 2, 1] = v[..., 0]
    
        return S
    
    def calc_M(self, R: torch.Tensor, V) -> torch.Tensor:
        """"""
        bs, _, _ = R.shape

        # Calc V(q)
        # if not R.requires_grad:
        #     R.requires_grad = True
        
        q = R.reshape(bs, 9)
        V_q = V(q)

        # Calc gradient 
        dV =  torch.autograd.grad(V_q.sum(), q, create_graph=True)[0]
        dV = dV.reshape(bs, 3, 3)

        # Calc skew(M) and extract M
        SM = torch.bmm(torch.transpose(dV, -2, -1), R) - torch.bmm(torch.transpose(R, -2, -1), dV)
        M = torch.stack((SM[..., 2, 1], SM[..., 0, 2], SM[..., 1, 0]), dim=-1).float()
        return M

    def cayley_transx(self, fc: torch.Tensor):
        """
        """
        F = torch.einsum('bij, bjk -> bik', (torch.eye(3, dtype=torch.float64, device=fc.device) + self.skew(fc)), torch.linalg.inv(torch.eye(3, dtype=torch.float64, device=fc.device) - self.skew(fc)))
        return F
    
    def calc_fc_init(self, a_vec: torch.Tensor, moi:torch.Tensor) -> torch.Tensor:
        """
        """
        fc_init = torch.einsum('bij, bj -> bi', torch.linalg.inv(2 * moi - self.skew(a_vec)), a_vec)
        return fc_init
    
    def calc_Ac(self, a_vec: torch.Tensor, moi: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        """
        
        Ac = a_vec + torch.einsum('bij, bj -> bi', self.skew(a_vec), fc) + torch.einsum('bj, b -> bj', fc, torch.einsum('bj, bj -> b', a_vec, fc)) - (2 * torch.einsum('ij, bj -> bi', moi, fc))
        return Ac
        
    def calc_grad_Ac(self, a_vec: torch.Tensor, moi: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        """
        grad_Ac = self.skew(a_vec) + torch.einsum('b, bij -> bij', torch.einsum('bi, bi -> b', a_vec, fc), torch.unsqueeze(torch.eye(3, dtype=torch.float64, device=a_vec.device), 0).repeat(fc.shape[0], 1, 1)) + torch.einsum('bi, bj -> bij', fc, a_vec) - (2 * moi)
        return grad_Ac
    
    def optimize_fc(self, R_vec: torch.Tensor, pi_vec: torch.Tensor, moi: torch.Tensor, V, fc_list: list = [], timestep: float = 1e-3, max_iter: int = 100, tol: float = 1e-8) -> list:
        """
        """
        it = 0
        M_vec = self.calc_M(R=R_vec, V=V)

        if not fc_list:
            a_vec = timestep * (pi_vec + (0.5 * timestep) * M_vec)
            fc_list.append(self.calc_fc_init(a_vec=a_vec, moi=moi))
        
        eps = torch.ones(fc_list[-1].shape[0], dtype=torch.float64)
        
        while  torch.any(eps > tol) and it < max_iter:
            
            fc_i = fc_list[-1]
            a_vec = timestep * (pi_vec + (0.5 * timestep) * M_vec)
            
            Ac = self.calc_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
            grad_Ac = self.calc_grad_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
           
            fc_ii = fc_i - torch.einsum('bij, bj -> bi', torch.linalg.inv(grad_Ac),  Ac)
            
            eps = torch.linalg.norm(fc_ii - fc_i, axis=-1)
            fc_list.append(fc_ii)
            it += 1
            
        return fc_list
    
    def step(self, R_i: torch.Tensor, pi_i: torch.Tensor, moi: torch.Tensor, V, fc_list: list = [], timestep: float = 1e-3):
        """
        """
        fc_list = self.optimize_fc(R_vec=R_i, pi_vec=pi_i, moi=moi, timestep=timestep, fc_list=fc_list, V=V)
        
        fc_opt = fc_list[-1]
        F_i = self.cayley_transx(fc=fc_opt)
        R_ii = torch.einsum('bij, bjk -> bik', R_i, F_i)
        
        M_i = self.calc_M(R=R_i, V=V)
        M_ii = self.calc_M(R=R_ii, V=V)
        pi_ii = torch.einsum('bji, bj -> bi', F_i, pi_i) + torch.einsum('bji, bj -> bi', 0.5 * timestep * F_i, M_i) + (0.5 * timestep) * M_ii
        
        return R_ii, pi_ii, fc_list
    
    def integrate(self, pi_init: torch.Tensor, R_init: torch.Tensor, moi: torch.Tensor, V, timestep: float = 1e-3, traj_len: int = 100):
        """
        """
        pi_list = [pi_init.double()]
        R_list = [R_init.double()]
        
        for it in range(1, traj_len):
            fc_list = []
            R_i = R_list[-1]
            pi_i = pi_list[-1]
            
            R_ii, pi_ii, fc_list = self.step(R_i=R_i, pi_i=pi_i, moi=moi, V=V, fc_list=fc_list, timestep=timestep)
            
            R_list.append(R_ii)
            pi_list.append(pi_ii)
        
        R_traj = torch.stack(R_list, axis=1)
        pi_traj = torch.stack(pi_list, axis=1)
        return R_traj, pi_traj

class LieGroupVaritationalIntegratorGeneral():
    """
    Lie group variational integrator with gavity.

    ...

    """
    def __init__(self):
        super().__init__()
        
    def skew(self, v: torch.Tensor):
        
        S = torch.zeros([v.shape[0], 3, 3], dtype=torch.float64, device=v.device)
        S[:, 0, 1] = -v[..., 2]
        S[:, 1, 0] = v[..., 2]
        S[:, 0, 2] = v[..., 1]
        S[:, 2, 0] = -v[..., 1]
        S[:, 1, 2] = -v[..., 0]
        S[:, 2, 1] = v[..., 0]
    
        return S
    
    def calc_M(self, R: torch.Tensor, V) -> torch.Tensor:
        """
        Calculate moments.

        ...

        Parameters
        ----------
        R:: torch.Tensor
            input rotation matrix

        V:: torch.nn.Module
            gravitational potential function -- most likely a neural network

        Returns
        -------
        M::torch.Tensor
            gravitational moment

        """
        # Calculate gravitational potential value
        bs, _, _ = R.shape
        q = R.reshape(bs, 9)
        V_q = V(q)

        # Calculate gradient of potential 
        dV =  torch.autograd.grad(V_q.sum(), q, create_graph=True)[0]
        dV = dV.reshape(bs, 3, 3)

        # Calculate skew(M) and extract M
        SM = torch.bmm(torch.transpose(dV, -2, -1), R) - torch.bmm(torch.transpose(R, -2, -1), dV)
        M = torch.stack((SM[..., 2, 1], SM[..., 0, 2], SM[..., 1, 0]), dim=-1).float()
        return M

    def cayley_transx(self, fc: torch.Tensor):
        """
        Calculate the Cayley transform.

        ...

        Parameter
        ---------
        fc:: torch.Tensor
            fc value

        Return
        ------
        F:: torch.Tensor
            F value

        """
       
        F = torch.einsum('bij, bjk -> bik', (torch.eye(3, dtype=torch.float64, device=fc.device) + self.skew(fc)), torch.linalg.inv(torch.eye(3, dtype=torch.float64, device=fc.device) - self.skew(fc)))
        return F
    
    def calc_fc_init(self, a_vec: torch.Tensor, moi:torch.Tensor) -> torch.Tensor:
        """
        Calculate the initial value fc.

        ...

        Parameter
        ---------
        a_vec :: torch.Tensor

        moi :: torch.Tensor 
            Moment-of-inertia tensor -- shape (3, 3).

        Return
        ------
        fc_init :: torch.Tensor
            iInitial value for fc

        """
        
        fc_init = torch.einsum('bij, bj -> bi', torch.linalg.inv(2 * moi - self.skew(a_vec)), a_vec)
        return fc_init
    
    def calc_Ac(self, a_vec: torch.Tensor, moi: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        Calculate the initial value fc.

        ...

        Parameter
        ---------
        a_vec::torch.Tensor
            
        moi::torch.Tensor 
            moment-of-inertia tensor

        fc::torch.Tensor
            fc tensor

        Return
        ------
        Ac::torch.Tensor
            Value of Ac

        """
        
        Ac = a_vec + torch.einsum('bij, bj -> bi', self.skew(a_vec), fc) + torch.einsum('bj, b -> bj', fc, torch.einsum('bj, bj -> b', a_vec, fc)) - (2 * torch.einsum('ij, bj -> bi', moi, fc))
        return Ac
        
    def calc_grad_Ac(self, a_vec: torch.Tensor, moi: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        Calculate the gradient of Ac.

        ...

        Parameters
        ----------
        a_vec :: torch.Tensor
            Vector -- shape (bs, 3)

        moi :: torch.Tensor
            Moment-of-inertia matrix -- shape (3, 3).

        fc :: torch.Tensor
            fc value -- shape (bs, 3).

        Returns
        -------
        grad_Ac :: torch.Tensor
            Gradient of Ac matrix -- shape (bs, 3, 3).

        """
        grad_Ac = self.skew(a_vec) + torch.einsum('b, bij -> bij', torch.einsum('bi, bi -> b', a_vec, fc), torch.unsqueeze(torch.eye(3, dtype=torch.float64, device=a_vec.device), 0).repeat(fc.shape[0], 1, 1)) + torch.einsum('bi, bj -> bij', fc, a_vec) - (2 * moi)
        return grad_Ac
    
    def optimize_fc(self, R_vec: torch.Tensor, pi_vec: torch.Tensor, moi: torch.Tensor, V = None, fc_list: list = [], timestep: float = 1e-3, max_iter: int = 5, tol: float = 1e-8) -> list:
        """
        Optimize the fc value.
        
        ...

        Parameters
        ----------
        R_vec :: torch.Tensor
            Orientation vector -- shape (bs, 3).
        
        pi_vec :: torch.Tensor
            Angular momentum vector -- shape (bs, 3).

        moi :: torch.Tensor
            Moment-of-inertia matrix -- shape (3, 3).

        V :: torch.nn.Module, default=None
            Potential energy function -- should be either a torch modulue or lambda function.

        fc_list :: list, default=[]
            List of fc values.

        timestep:: torch.Tensor, defualt=1e-3
            Timestep value used during integration.

        max_iter :: int, default=5
            Maximum number of iterations to use for the Newton Raphson method (NRM).

        tol :: float, default=1e-8
            Tolerance to exit the NRM loop.

        Returns
        -------
        fc_list::torch.Tensor

        """
        # Count then number of iterations
        it = 0

        # If there is no potential, no moment due to potential
        if V is None:
            M_vec = torch.zeros_like(pi_vec, dtype=torch.float64)
        else:
            M_vec = self.calc_M(R=R_vec, V=V)

        # Initialize fc
        if not fc_list:
            a_vec = timestep * (pi_vec + (0.5 * timestep) * M_vec)
            fc_list.append(self.calc_fc_init(a_vec=a_vec, moi=moi))
        
        eps = torch.ones(fc_list[-1].shape[0], dtype=torch.float64)
        
        # Optimization loop -- Newton Raphson method
        while  torch.any(eps > tol) and it < max_iter:
            
            fc_i = fc_list[-1]
            a_vec = timestep * (pi_vec + (0.5 * timestep) * M_vec)
            
            Ac = self.calc_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
            grad_Ac = self.calc_grad_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
           
            fc_ii = fc_i - torch.einsum('bij, bj -> bi', torch.linalg.inv(grad_Ac),  Ac)
            
            eps = torch.linalg.norm(fc_ii - fc_i, axis=-1)
            fc_list.append(fc_ii)
            it += 1
            
        return fc_list
    
    def step(self, R_i: torch.Tensor, pi_i: torch.Tensor, moi: torch.Tensor, u_i: torch.Tensor = None, u_ii: torch.Tensor = None, V = None, fc_list: list = [], timestep: float = 1e-3):
        """
        Calculate next step using the dynamics and kinematics.

        ...

        Parameters
        ----------
       pi_i :: torch.Tensor
            Initial condition for angular momentum vector -- shape (batch size, 3).

        R_i :: torch.Tensor
            Intial condition for orientation matrix  -- shape (batch size, 3, 3).

        moi :: torch.Tensor
            Moment-of-inertia tensor -- shape (3, 3).

        u_i :: torch.Tensor, default=None
            Control moment input for timestep i -- shape (batch size, 3).

        u_ii :: torch.Tensor, default=None
            Control moment input for timestep ii -- shape (batch size, 3).

        V :: torch.nn.Module, default=None
            Potential energy function -- should be either a torch modulue or lambda function.

        fc_list :: list, default=[]
            fc list

        timestep :: float, defualt=1e-3
            Timestep used for integration.

        Returns
        -------
        pi_ii :: torch.Tensor
            Angular momentum vector for next timestep -- shape (batch size, 3).

        R_ii :: torch.Tensor
            Orientation matrix for next timestep  -- shape (batch size, 3, 3).
        
        fc_list :: list

        """
        # Calculate list of optimal fc
        fc_list = self.optimize_fc(R_vec=R_i, pi_vec=pi_i, moi=moi, timestep=timestep, fc_list=fc_list, V=V)
        
        # Selected optimal fc
        fc_opt = fc_list[-1]

        # Update pose using kinematics
        F_i = self.cayley_transx(fc=fc_opt)
        R_ii = torch.einsum('bij, bjk -> bik', R_i, F_i)
        
        # Calculate moment due to potential function
        if V is None:
            M_i = torch.zeros_like(pi_i, dtype=torch.float64)
            M_ii = torch.zeros_like(pi_i, dtype=torch.float64)
        else:
            M_i = self.calc_M(R=R_i, V=V)
            M_ii = self.calc_M(R=R_ii, V=V)
        
        # Grab control moment
        if u_i is None:
            u_i = torch.zeros_like(pi_i, dtype=torch.float64)

        if u_ii is None:
            u_ii = torch.zeros_like(pi_i, dtype=torch.float64)
        
        # Update angular momentum state
        pi_ii = torch.einsum('bji, bj -> bi', F_i, pi_i) + torch.einsum('bji, bj -> bi', 0.5 * timestep * F_i, u_i + M_i) + (0.5 * timestep) * (M_ii + u_ii)
        
        return R_ii, pi_ii, fc_list
    
    def integrate(self, pi_init: torch.Tensor, R_init: torch.Tensor, moi: torch.Tensor, u_control: torch.Tensor = None, V = None, timestep: float = 1e-3, traj_len: int = 100):
        """
        Method to integrate a full trajectory.
        ...

        Parameters
        ----------
        pi_init :: torch.Tensor
            Initial condition for angular momentum vector -- shape (batch size, 3).

        R_init :: torch.Tensor
            Intial condition for orientation matrix  -- shape (batch size, 3, 3).

        moi :: torch.Tensor
            Moment-of-inertia tensor -- shape (3, 3).

        u_control :: torch.Tensor, default=None
            Control input tensor -- shape (batch size, trajectory length, 3).

        V :: torch.nn.Module, default=None
            Potential energy function -- should be either a torch modulue or lambda function.

        timestep :: float, defualt=1e-3
            Timestep used for integration.

        traj_len :: int, default=100
            Trajectory length of the full trajectory.
        
        Returns
        -------
        R_traj :: torch.Tensor
        pi_traj :: torch.Tensor

        """
        pi_list = [pi_init.double()]
        R_list = [R_init.double()]
        
        # Integrate full trajectory
        for it in range(1, traj_len):
            # Initialize inputs 
            fc_list = []
            R_i = R_list[-1]
            pi_i = pi_list[-1]

            # Control moment
            if u_control is None:
                u_i = torch.zeros_like(pi_i, dtype=torch.float64)
                u_ii = torch.zeros_like(pi_ii, dtype=torch.float64)
            else:
                u_i = u_control[:, it-1, ...]
                u_ii = u_control[:, it, ...]

            # Calculate next timestep
            R_ii, pi_ii, fc_list = self.step(R_i=R_i, pi_i=pi_i, moi=moi, u_i=u_i, u_ii=u_ii, V=V, fc_list=fc_list, timestep=timestep)
            
            # Append to state lists
            R_list.append(R_ii)
            pi_list.append(pi_ii)
        
        # Append full trajectory together
        R_traj = torch.stack(R_list, axis=1)
        pi_traj = torch.stack(pi_list, axis=1)
        return R_traj, pi_traj
    
def build_V_gravity(m: float,
                    R: torch.Tensor, 
                    e_3: torch.Tensor, 
                    rho_gt: torch.Tensor,
                    g: float = 9.81):
  """
  Potential energy function for gravity.

  ...

  Parameters
  ----------
  m : [1]
    mass of object
  g :  [1]
    gravity constant
  e_3 : [3]
    z-direction
  R : [bs, 3, 3]
    Rotation matrix
  rho_gt : 

  """
  bs = R.shape[0]
  R_reshape = R.reshape(bs, 3, 3)
  R_times_rho_gt = torch.einsum('bij, j -> bi', R_reshape, rho_gt.squeeze())
  e3T_times_R_times_rho_gt = torch.einsum('j, bj -> b', e_3.squeeze(), R_times_rho_gt)
  Vg = -m * g * e3T_times_R_times_rho_gt
  
  return Vg

# Functions for generating datasets for freely rotating RBD
def euler_eigvec(MOI: np.ndarray, radius: float) -> np.ndarray:
    """
    Function to calculate the eigenvectors of the Euler dynamics, linearized about the intermediate axis.
    
    ...
    
    Parameters
    ----------
    MOI : np.ndarray
        Moment of intertia tensor for the system.
        
    radius : float
        Radius of the angular momentum sphere.
    
    Returns
    -------
    eigvec : np.ndarray
        Eigenvectors correpsonding to the dynamics after they're linearized about the intermediate axis.
        
    Notes
    -----
    
    """
    MOI = np.diag(MOI)
    beta = (MOI[0] - MOI[1])/(MOI[0] * MOI[1]) # factor used for linearization
    gamma = (MOI[1] - MOI[2])/(MOI[1] * MOI[2]) # factor used for linearization
    
    euler_umatrix = np.array([[0, 0, beta * radius], [0, 0, 0], [gamma * radius, 0 , 0]]) # linearize dyns
    _, eigvec = np.linalg.eig(euler_umatrix) # calculate the eigenvalues and eigenvectors 
    
    return eigvec

def calc_hetero_angle(eigv: np.ndarray) -> np.ndarray:
    """
    """
    e3 = np.array([0., 0., 1.]).reshape((3,))
    v1 = eigv[:, 0]
    
    # Calculate angle using the first eigenvalue and z-axis 
    cos_theta = np.max(np.min((np.dot(v1, e3)/(np.linalg.norm(v1) * np.linalg.norm(e3))), axis=0), axis=-1)
    angle = np.real(np.arccos(cos_theta))
    return angle

def sample_init_conds(MOI: np.ndarray, radius: float, seed: int = 0, ic_type: str = 'random', n_samples: int = 10, desired_samples: np.ndarray = None, bandwidth_us: float = 5.0) -> np.ndarray:
    """
    Function to sample from the body angular momentum sphere.
    
    ...
    
    """
    np.random.seed(seed=seed)
    ic_type = ic_type.lower()
    eps = 1e-3
    
    if ic_type == 'random':
        theta = np.random.uniform(low=0.0, high=np.pi+eps, size=(n_samples, 1))
        phi = np.random.uniform(low=0.0, high=2*np.pi, size=(n_samples, 1))
    
    elif ic_type == 'unstable':
        assert bandwidth_us < 10
        bw_rad = np.deg2rad(bandwidth_us)
        ev = euler_eigvec(MOI=MOI, radius=radius)
        heteroclinic_angle = calc_hetero_angle(eigv=ev)
        
        theta = np.random.uniform(low = heteroclinic_angle - (0.5 * bw_rad), high = heteroclinic_angle + (0.5 * bw_rad), size=(n_samples, 1))
        phi = np.zeros((n_samples, 1))
        
    elif ic_type =='desired':
        theta = desired_samples[0, ...]
        phi = desired_samples[1, ...]
    else:
        raise ValueError('Use the allowed ic_type.')
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    samples = np.concatenate((x, y, z), axis=-1)
    return samples

def generate_lowdim_dataset(MOI: np.ndarray, radius: float, n_samples: int, integrator, timestep: float = 1e-3, traj_len: int = 100, bandwidth_us: float = 5., desired_samples: np.ndarray = None, ic_type: str = 'random', V = None, seed: int = 0) -> np.ndarray:
    """"""
    # sample initial conditions 
    # body angular momentum sphere
    pi_samples = sample_init_conds(MOI=MOI, radius=radius, ic_type=ic_type, n_samples=n_samples, bandwidth_us=bandwidth_us)
    pi_samples_tensor = torch.tensor(pi_samples, device=MOI.device)
    
    # group element matrices
    R_samples = random_group_matrices(n=n_samples)

    # Integrate trajectories
    data_R, data_pi = integrator.integrate(pi_init=pi_samples_tensor, moi=MOI, R_init=R_samples, timestep=timestep, traj_len=traj_len)
    return data_R, data_pi

# Functions for data generation for 3D Pendulum and Physical Pendulum
def sample_group_matrices_3DP(radius: float, ic_type: str = 'uniform', n_samples: int = 100, scale: float = 0.1, specified_samples: np.ndarray = None):
    """"""
    ic_type = ic_type.lower()
    eps = 1e-3

    if ic_type == 'stable':
        stable  = np.array([0., 0., 0.])[None, ...].repeat(n_samples, 1)
        
        phi = np.random.uniform(low=-scale*np.pi, high=scale*np.pi, size=(n_samples, 1)) + stable[:, 0]
        theta = np.random.uniform(low=-0.5*scale*np.pi, high=0.5*scale*np.pi, size=(n_samples, 1)) + stable[:, 1]
        psi = np.random.uniform(low=-scale*np.pi, high=scale*np.pi, size=(n_samples, 1)) + stable[:, 2]

    elif ic_type == 'unstable':
        unstable  = np.array([0., np.pi, 0.])[None, ...].repeat(n_samples, 1)
        
        phi = np.random.uniform(low=-scale*np.pi, high=scale*np.pi, size=(n_samples, 1)) + unstable[:, 0]
        theta = np.random.uniform(low=-0.5*scale*np.pi, high=0.5*scale*np.pi, size=(n_samples, 1)) + unstable[:, 1]
        psi = np.random.uniform(low=-scale*np.pi, high=scale*np.pi, size=(n_samples, 1)) + unstable[:, 2]

    elif ic_type == 'uniform':
        phi = np.random.uniform(low=0.0, high=2*np.pi-eps, size=(n_samples, 1))
        theta = np.random.uniform(low=-0.5*scale*np.pi, high=0.5*scale*np.pi, size=(n_samples, 1))
        psi = np.random.uniform(low=-scale*np.pi, high=scale*np.pi, size=(n_samples, 1))

    else:
        raise ValueError(f"Use 'ic_type' from the allowed set: {['stable', 'unstable', 'uniform']}")

    phi_tensor = torch.tensor(phi)
    theta_tensor = torch.tensor(theta)
    psi_tensor = torch.tensor(psi)

    euler_angles = torch.cat((phi_tensor, theta_tensor, psi_tensor), dim=-1)

    R = rc.euler_angles_to_matrix(euler_angles=euler_angles, convention='ZXZ') # eazyz_to_group_matrix(alpha=phi, beta=theta, gamma=psi).squeeze()
    samples = R # .transpose(2, 0, 1)

    return samples

def sample_group_matrix_heavytop(args, MOI: np.ndarray, n_samples: int,  mass: float = 1.0, l: float = 1.0, general_flag: bool = False):
    """"""
    if general_flag:
        # Euler angles for gyroscope
        eulers = (np.random.rand(n_samples, 2, 3) - 0.5) * 3 #3

        # Edit euler angles to be in a desired range (should be psidot >> thetadot >> phidot)
        eulers[:,1,0]*=3 # phidot [magnitude 4.5]
        eulers[:,1,1]*=.2 # thetadot [magnitude 0.3]
        eulers[:,1,2] = (np.random.randint(2, size=(n_samples, )) * 2. - 1) * (np.random.randn(n_samples) + 7) * 1.5 # psidot [manigtude 7]

        # Assign Euler angles -- general
        phi = eulers[:, 0, 0]
        theta = eulers[:, 0, 1]
        psi = eulers[:, 0, 2]

        # Calculate omega in the body-fixed frame
        phidot = eulers[:, 1, 0]
        thetadot = eulers[:, 1, 1]
        psidot = eulers[:, 1, 2]

        w3 = (phidot * ct) + psidot

    else:

        # Assign Euler angles -- steady top
        g = 9.8 #gravity
        m = mass
        r = args.radius
        l3 = l

        I_t = (1/4.)*m*(r**2) + (m*(l3**2))
        I_a = (1/2.)*m*(r**2)

        I_1 = I_t
        I_3 = I_a

        eulers = (np.random.rand(n_samples, 3) - 0.5) * (np.pi - 0.1)

        phi = eulers[:, 0]
        theta = eulers[:, 1]
        psi = eulers[:, 2] 

        # Thetadot initialized to zero
        thetadot = np.zeros((n_samples))
    
        # Phidot samples from range
        w3_min = (2./I_3) * np.sqrt(mass * g * l * I_1 * np.cos(theta))
        w3 = np.einsum('b, b -> b', (np.random.rand((n_samples)) + 2.1), w3_min)

        phidot_slow = (mass * g * l)/ (I_3 * w3)  
        phidot_fast = (I_3 * w3)/(I_1 * np.cos(theta))

        # Psidot is calculated
        psidot = phidot_fast # phidot
        phidot = ((mass * g * l + (I_1 - I_3) * (psidot ** 2) * np.cos(theta))/(I_3 * psidot)) # psidot = f(phidot) 
        # import pdb; pdb.set_trace()
    

    st = np.sin(theta)
    ct = np.cos(theta)

    sp = np.sin(psi)
    cp = np.cos(psi)

    # Calculate pi in the body frame -- Goldstein 
    w1 = (phidot * st * sp) + (thetadot * cp)
    w2 = (phidot * st * cp) - (thetadot * sp)

    omega = np.stack([w1, w2, w3], axis=-1)
    pi_samples = np.einsum('ij, bj -> bi', MOI, omega)

    # Convert the Euler angles to rotation matrices using the ZYZ convention
    phi_tensor = torch.tensor(phi)
    theta_tensor = torch.tensor(theta)
    psi_tensor = torch.tensor(psi)

    euler_angles = torch.stack((phi_tensor, theta_tensor, psi_tensor), dim=-1)

    R_samples = rc.euler_angles_to_matrix(euler_angles=euler_angles, convention='ZXZ')
    return R_samples, pi_samples

def generate_lowdim_dataset_3DP(MOI: np.ndarray, radius: float, n_samples: int, integrator, timestep: float = 1e-3, traj_len: int = 100, bandwidth_us: float = 5., desired_samples: np.ndarray = None, R_ic_type: str = 'stable', pi_ic_type: str = 'random', V = None, seed: int = 0):
    """"""
    # sample initial conditions 
    # body angular momentum sphere
    pi_samples = sample_init_conds(MOI=MOI, radius=radius, ic_type=pi_ic_type, n_samples=n_samples, bandwidth_us=bandwidth_us)
    pi_samples_tensor = torch.tensor(pi_samples, device=MOI.device)
    
    # group element matrices
    R_samples = sample_group_matrices_3DP(radius=radius, ic_type=R_ic_type, n_samples=n_samples)
    R_samples_tensor = torch.tensor(R_samples, device=MOI.device, requires_grad=True)
    
    # integrate trajectories
    data_R, data_pi = integrator.integrate(pi_init=pi_samples_tensor, moi=MOI, V=V, R_init=R_samples_tensor, timestep=timestep, traj_len=traj_len)
    return data_R, data_pi

def generate_lowdim_dataset_heavytop(args, MOI: np.ndarray, mass: float, l3: float, n_samples: int, integrator, general_flag: bool = False, timestep: float = 1e-3, traj_len: int = 100, V = None, seed: int = 0):
    """"""
    # Sample initial conditions
    R_samples, pi_samples = sample_group_matrix_heavytop(args=args, MOI=MOI, n_samples=n_samples, mass=mass, l=l3, general_flag=False)

    # Make samples tensors
    R_samples_tensor = R_samples.clone().detach().requires_grad_(True)
    pi_samples_tensor = torch.tensor(pi_samples)

    # Integrate trajectories
    data_R, data_pi = integrator.integrate(pi_init=pi_samples_tensor, moi=MOI, V=V, R_init=R_samples_tensor, timestep=timestep, traj_len=traj_len)

    return data_R, data_pi

# Auxilary function for dataset generation
def random_quaternions(n, dtype=torch.float32, device=None):
    u1, u2, u3 = torch.rand(3, n, dtype=dtype, device=device)
    return torch.stack((
        torch.sqrt(1-u1) * torch.sin(2 * np.pi * u2),
        torch.sqrt(1-u1) * torch.cos(2 * np.pi * u2),
        torch.sqrt(u1) * torch.sin(2 * np.pi * u3),
        torch.sqrt(u1) * torch.cos(2 * np.pi * u3),
    ), 1)

def quaternions_to_group_matrix(q):
    """Normalises q and maps to group matrix."""
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack([
        r*r - i*i - j*j + k*k, 2*(r*i + j*k), 2*(r*j - i*k),
        2*(r*i - j*k), -r*r + i*i - j*j + k*k, 2*(i*j + r*k),
        2*(r*j + i*k), 2*(i*j - r*k), -r*r - i*i + j*j + k*k
    ], -1).view(*q.shape[:-1], 3, 3)

def random_group_matrices(n, dtype=torch.float32, device=None):
    return quaternions_to_group_matrix(random_quaternions(n, dtype, device))

def make_gif(frame_folder, gif_name: str, save_dir: str):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*"))]
    frame_one = frames[0]
    # print(f'\n List of sorted frames: {print(sorted(glob.glob(f"{frame_folder}/*")))} \n')
    frame_one.save(f"{save_dir}/{gif_name}.gif", format="GIF", append_images=frames[1:],
               save_all=True, duration=100, loop=0, optimize=False)
