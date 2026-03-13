import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y):
        super(ExactGPModel, self).__init__(train_x, train_y, gpytorch.likelihoods.GaussianLikelihood())

        dim = train_x.shape[1]

        self.mean_module = gpytorch.means.LinearMean(input_size = dim)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = dim))

        self.covar_module.base_kernel.lengthscale = torch.rand(dim) * 500 + 800
        self.covar_module.outputscale = 1.0

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)