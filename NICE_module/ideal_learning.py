import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import interpolate
from torchdiffeq import odeint

# Dictionary of activation functions
activations = {'relu': torch.nn.ReLU(),
               'sigmoid': torch.nn.Sigmoid(),
               'elu': torch.nn.ELU(),
               'tanh': torch.nn.Tanh(),
               'gelu': torch.nn.GELU(),
               'silu': torch.nn.SiLU(),
               'softplus': torch.nn.Softplus(beta=1, threshold=20),
               'leaky_relu': torch.nn.LeakyReLU()}

class NICE(torch.nn.Module):
    '''
    Neural Integration for Constitutive Equations (NICE)
    '''

    def __init__(self, params_evolution, params_energy, number_IC, norm_params, dim=2, dtype=torch.float32):
        super(NICE, self).__init__()

        # Set data type and dimension
        self.dtype = dtype
        self.dim = dim

        # Unpack normalization parameters
        self.prm_e, self.prm_de, self.prm_s, self.prm_dt = norm_params

        # Calculate fractions for el and de normalization
        frac = 0.5
        self.prm_ee = self.prm_e * frac
        self.prm_dee = self.prm_de * frac

        # Initialize solver and neural networks for evolution and energy
        self.solver = None
        self.NeuralNetEvolution = self.constructor(params_evolution)
        self.NeuralNetEnergy = self.constructor(params_energy)
        self.relu = torch.nn.ReLU()

        # Initialize elastic strain parameter and normalization factor
        self.e0 = torch.nn.Parameter(torch.zeros((number_IC, self.dim)), requires_grad=True)
        self.prm_u = np.linalg.norm(self.prm_s, axis=1) * np.linalg.norm(self.prm_e, axis=1)
        self.inference = None

    def constructor(self, params):
        '''
        Feed-forward artificial neural network constructor
        :params : [input layer # nodes, output layer # nodes, hidden layers # node, hidden activations]
        '''
        i_dim, o_dim, h_dim, act = params
        dim = i_dim
        layers = torch.nn.Sequential()
        for hdim in h_dim:
            layers.append(torch.nn.Linear(dim, hdim, dtype=self.dtype))
            layers.append(activations[act])
            dim = hdim
        layers.append(torch.nn.Linear(dim, o_dim, dtype=self.dtype))
        return layers

    def Normalize(self, inputs, prm):
        '''
        Normalize features
        :inputs : data
        :prm : normalization parameters
        '''
        return torch.divide(torch.add(inputs, -prm[1]), prm[0])

    def DeNormalize(self, outputs, prm):
        '''
        Denormalize features
        :output : dimensionless data
        :prm : normalization parameters
        '''
        return torch.add(torch.multiply(outputs, prm[0]), prm[1])

    def forward(self, t, y):
        # Extract elastic strain and normalize
        uel = y[:, :self.dim]
        nel = self.Normalize(uel, self.prm_ee)

        # Determine total strain rate (ueps_dot)
        if self.inference == False:
            if t > 1.:
                ueps_dot = self.eps_dot[-1]
            else:
                ueps_dot = self.eps_dot[int(t / self.prm_dt)]
        else:
            # Interpolate external data for inference
            ueps_dot = torch.zeros((len(self.idx), 2))
            for i in range(len(self.idx)):
                ueps_dot[i, 0] = torch.from_numpy(self.interp_dotev[self.idx[i]](t.detach().numpy()))
                ueps_dot[i, 1] = torch.from_numpy(self.interp_dotes[self.idx[i]](t.detach().numpy()))
        neps_dot = self.Normalize(ueps_dot, self.prm_de).detach()

        # Feed-forward neural network for evolution
        nodes = self.NeuralNetEvolution(torch.cat((neps_dot, nel), -1))
        node_el = nodes[:, :self.dim]

        # De-normalize the output
        uode_el = self.DeNormalize(node_el, self.prm_dee)

        return uode_el

    def stress(self, uel):
        # Normalize elastic strain
        nel = self.Normalize(uel, self.prm_ee)

        # Neural network for energy
        nu = self.NeuralNetEnergy(nel)

        # De-normalize energy
        u = self.DeNormalize(nu, self.prm_u)

        # Calculate stress
        ustress = (torch.autograd.grad(u, uel, grad_outputs=torch.ones_like(u),
                                      retain_graph=True, create_graph=True)[0])
        return ustress

    def integrate(self, u, y0, t, idx):
        # Integrate the ODE using torchdiffeq
        self.eps_dot = u
        self.idx = idx
        y_ = odeint(self, y0, t, method=self.solver, options={"step_size": self.step_size})

        # Extract normalized variables
        uel = y_[:, :, :self.dim]
        nel = self.Normalize(uel, self.prm_ee)

        # Calculate stress
        stress = self.stress(uel)
        nstress = self.Normalize(stress, self.prm_s)
        neps_dot = self.Normalize(self.eps_dot, self.prm_de)

        # Feed-forward neural network for evolution
        nodes = self.NeuralNetEvolution(torch.cat((neps_dot, nel), -1))
        node_el = nodes[:, :, :self.dim]

        # De-normalize the output
        uode_el = self.DeNormalize(node_el, self.prm_dee)

        # Calculate the dissipation rate
        uode_pl = self.eps_dot - uode_el
        sijdepij = torch.einsum('ijk,ijk->ij', stress[1:], uode_pl[:-1])

        return y_, stress, sijdepij

    def init_interp(self, args, t):
        # Initialize interpolation for external data
        self.x = np.arange(args.shape[1])
        self.interp_dotev = []
        self.interp_dotes = []
        for i in range(len(self.x)):
            f = interpolate.interp1d(t, args[:, i, 0], fill_value="extrapolate", kind="previous")
            g = interpolate.interp1d(t, args[:, i, 1], fill_value="extrapolate", kind="previous")
            self.interp_dotev.append(f)
            self.interp_dotes.append(g)

    def find_elastic_strain(self, eps_e, sigma):
        # Find elastic strain using a root-finding method
        eps_e_tensor = torch.from_numpy(eps_e.reshape(-1, 2))
        eps_e_tensor.requires_grad = True
        ueps_e_tensor = self.DeNormalize(eps_e_tensor, self.prm_ee)
        ustress = self.stress(ueps_e_tensor)
        rhs = self.Normalize(sigma, self.prm_s).detach() - self.Normalize(ustress, self.prm_s).detach().numpy()
        return rhs.reshape(-1)

def slice_data(x, ntrainval, ntest):
    """
    Slices the data into training and testing sets.

    Args:
        x: Input data.
        ntrainval (int): Number of samples for training and validation.
        ntest (int): Number of samples for testing.

    Returns:
        Tuple: Sliced training/validation data, sliced testing data.
    """
    return x[:, ntrainval], x[:, ntest]

def get_params(x, norm=False, vectorial_norm=False):
    '''
    Compute normalization parameters:
        - normalize ([-1,1]) component by component (vectorial_norm = True)
        - normalize ([-1,1]) (vectorial_norm = False, norm = True)
        - standardize (vectorial_norm = False, norm = False)

    Args:
        x: Input data.
        norm (bool): Normalize data to [-1,1].
        vectorial_norm (bool): Normalize data component by component (along axis = 1).

    Returns:
        torch.Tensor: Normalization parameters.
    '''
    if vectorial_norm == False:
        if norm == True:
            # Normalize to [-1, 1]
            A = 0.5 * (np.amax(x) - np.amin(x))
            B = 0.5 * (np.amax(x) + np.amin(x))
        else:
            # Standardize (mean = 0, std = 1)
            A = np.std(x, axis=(0, 1))
            B = np.mean(x, axis=(0, 1))
    else:
        # Normalize component by component (along axis = 1)
        dim = x.shape[-1]
        u_max = np.zeros((dim,))
        u_min = np.zeros((dim,))
        for i in np.arange(dim):
            u_max[i] = np.amax(x[:, i])
            u_min[i] = np.amin(x[:, i])
        A = (u_max - u_min) / 2.
        B = (u_max + u_min) / 2.
        A[A == 0] = 1
    return torch.tensor(np.array([np.float64(A), np.float64(B)]))