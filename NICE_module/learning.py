# Import necessary libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy import interpolate
from torchdiffeq import odeint  # Neural ODE solver

# Dictionary of common activation functions used in neural networks
activations = {
    'relu': torch.nn.ReLU(),
    'sigmoid': torch.nn.Sigmoid(),
    'elu': torch.nn.ELU(),
    'tanh': torch.nn.Tanh(),
    'gelu': torch.nn.GELU(),
    'silu': torch.nn.SiLU(),
    'softplus': torch.nn.Softplus(beta=5, threshold=20),
    'leaky_relu': torch.nn.LeakyReLU()
}

class NICE(torch.nn.Module):
    """
    Neural Integration for Constitutive Equations (NICE)
    This model leverages neural networks to approximate constitutive equations
    governing stress-strain relationships in materials.
    """

    def __init__(self, params_f, params_ψ, number_IC, norm_params, dim=2, dtype=torch.float64):
        """
        Initializes the NICE model.
        
        :param params_f: Network architecture for the evolution equations
        :param params_ψ: Network architecture for the energy function
        :param number_IC: Number of initial conditions considered
        :param norm_params: Normalization parameters (strain, density, stress)
        :param dim: Dimensionality of the problem (default is 2D)
        :param dtype: Data type for PyTorch tensors (default: torch.float64)
        """
        super(NICE, self).__init__()

        # Set model attributes
        self.dtype = dtype
        self.dim = dim

        # Unpack normalization parameters
        self.prm_strain, self.prm_density, self.prm_stress = norm_params

        # Initialize a normalization factor for strain decomposition
        self.frac = 1
        self.prm_strain_elastic = self.prm_strain * self.frac

        # Define neural networks for evolution and energy functions
        self.NeuralNetEvolution = self.constructor(params_f)
        self.NeuralNetEnergy = self.constructor(params_ψ)

        # Activation function for enforcing positivity (Macaulay brackets)
        self.MacaulayBrackets = torch.nn.ReLU()

        # Define learnable initial elastic strain
        self.initial_elastic_strain = torch.nn.Parameter(torch.zeros((number_IC, self.dim)), requires_grad=True)

        # Compute normalization factor for energy
        self.prm_energy = np.linalg.norm(self.prm_stress, axis=1) * np.linalg.norm(self.prm_strain, axis=1)
        self.prm_energy /= self.prm_density[0]

    def constructor(self, params):
        """
        Constructs a feed-forward artificial neural network.
        
        :param params: A list specifying:
            - Input layer size
            - Output layer size
            - Hidden layer sizes
            - Activation function type
        :return: A PyTorch sequential model
        """
        i_dim, o_dim, h_dim, act = params
        dim = i_dim  # Set input dimension

        # Initialize a sequential model
        layers = torch.nn.Sequential()

        # Define hidden layers with activation functions
        for hdim in h_dim:
            layers.append(torch.nn.Linear(dim, hdim, dtype=self.dtype))
            layers.append(activations[act])  # Select activation function
            dim = hdim

        # Define output layer
        layers.append(torch.nn.Linear(dim, o_dim, dtype=self.dtype))

        return layers

    def Normalize(self, inputs, prm):
        """
        Normalizes input features using stored parameters.
        
        :param inputs: Raw input data
        :param prm: Normalization parameters
        :return: Normalized data
        """
        return torch.divide(torch.add(inputs, -prm[1]), prm[0])

    def DeNormalize(self, outputs, prm):
        """
        Denormalizes features to retrieve physical values.
        
        :param outputs: Normalized data
        :param prm: Normalization parameters
        :return: Original scale data
        """
        return torch.add(torch.multiply(outputs, prm[0]), prm[1])

    def forward(self, t, state):
        """
        Defines the forward pass of the model for time-dependent evolution.
        
        :param t: Time step
        :param state: Current state of the system
        :return: Evolution of elastic strain, density, and dissipation rate
        """
        # Extract and normalize elastic strain
        elastic_strain = state[:, :self.dim]
        norm_elastic_strain = self.Normalize(elastic_strain, self.prm_strain_elastic)

        # Compute total strain rate via interpolation
        strain_rate = torch.zeros((elastic_strain.shape[0], 2))
        strain_rate[:, 0] = torch.from_numpy(self.interp_dotev(t.detach().numpy(), self.idx, grid=False))
        strain_rate[:, 1] = torch.from_numpy(self.interp_dotes(t.detach().numpy(), self.idx, grid=False))
        norm_strain_rate = self.Normalize(strain_rate, self.prm_strain_rate)
        
        # Extract and normalize density
        density = state[:, self.dim:self.dim + 1]
        norm_density = self.Normalize(density, self.prm_density)

        # Compute stress from the current state
        stress_e, p_T, stress = self.stress([elastic_strain, density], grads=True)
        norm_stress = self.Normalize(stress_e, self.prm_stress)

        # Compute rate of change of density (mass balance)
        density_rate = density * strain_rate[:, :1]

        # Neural network prediction for plastic strain rate evolution
        ODE = self.NeuralNetEvolution(torch.cat((norm_stress, norm_density), -1))

        # Compute plastic and elastic strain rates
        norm_elastic_strain_rate = ODE[:, :self.dim]
        elastic_strain_rate = self.DeNormalize(norm_elastic_strain_rate, self.prm_strain_rate)
        plastic_strain_rate = strain_rate - elastic_strain_rate

        # Compute dissipation rate
        dissipation_rate = torch.sum(stress * plastic_strain_rate, dim=-1)[:, None]

        return torch.cat((elastic_strain_rate, density_rate, dissipation_rate), -1)

    def stress(self, X, grads=False):
        """
        Computes stress based on elastic strain and density.
        
        :param X: Tuple containing (elastic_strain, density)
        :param grads: If True, computes gradients for backpropagation
        :return: Stress tensor
        """
        elastic_strain, density = X

        # Normalize inputs
        norm_elastic_strain = self.Normalize(elastic_strain, self.prm_strain_elastic)
        norm_density = self.Normalize(density, self.prm_density)

        # Compute energy from neural network
        svars = torch.cat((norm_elastic_strain, norm_density), -1)
        norm_energy = self.NeuralNetEnergy(svars)
        energy = self.DeNormalize(norm_energy, self.prm_energy)

        # Compute stress as derivative of energy w.r.t strain
        stress_e = density * (torch.autograd.grad(energy, elastic_strain,
                                                grad_outputs=torch.ones_like(energy),
                                                create_graph=True,
                                                retain_graph=True)[0])
        chemical_potential = density* (torch.autograd.grad(energy, density,
                                                           grad_outputs=torch.ones_like(energy),
                                                           create_graph=True,
                                                           retain_graph=True)[0])
        stress_pT = torch.cat((chemical_potential * density, torch.zeros(energy.shape)), -1)
        return stress_e, stress_pT, stress_e+stress_pT

    def integrate(self, initial_conditions, t, idx):
        """
        Solves the ODE system using a numerical integrator.
        
        :param initial_conditions: Initial state variables
        :param t: Time array
        :param idx: Protocol index for interpolation
        :return: Integrated state variables and computed stress
        """
        self.idx = idx  # Store protocol index for interpolation

        solution = odeint(self, initial_conditions, t, method=self.solver,
                          rtol=self.rtol, atol=self.atol)

        # Extract relevant state variables
        elastic_strain = solution[:, :, :self.dim]
        density = solution[:, :, self.dim:self.dim + 1]
        dissipation = solution[:, :, self.dim + 1:]

        # Compute stress
        stress_e, p_T, stress = self.stress([elastic_strain, density], grads=True)

        return solution, stress, dissipation


    def init_interp(self, args, t):
        """
        Initializes interpolation for strain rate values.
        """
        len_ = np.arange(args.shape[1])
        self.interp_ev = RectBivariateSpline(t, len_, args[:, :, 0], kx=2, ky=2, s=0)
        self.interp_es = RectBivariateSpline(t, len_, args[:, :, 1], kx=2, ky=2, s=0)
        self.interp_dotev = self.interp_ev.partial_derivative(dx=1, dy=0)
        self.interp_dotes = self.interp_es.partial_derivative(dx=1, dy=0)

    def find_elastic_strain(self, x, state):
        """
        Solves for elastic strain using a root-finding method.
        """
        density, stress = state
        x = torch.from_numpy(x.reshape(-1, 2))
        x.requires_grad = True
        elastic_strain = self.DeNormalize(x, self.prm_strain_elastic)
        stress_e, p_T, predicted_stress = self.stress([elastic_strain, density])

        rhs = self.Normalize(stress, self.prm_stress).detach() - self.Normalize(predicted_stress, self.prm_stress).detach().numpy()
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
            if type(A) != np.float64:
                A[A == 0] = 1
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