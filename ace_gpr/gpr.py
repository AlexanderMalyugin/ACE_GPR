import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self,
                 train_x,
                 train_y,
                 initial_kernel_parameters,
                 likelihood
                 ):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        dim = train_x.shape[1]

        self.mean_module = gpytorch.means.LinearMean(input_size = dim)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = dim))

        self.covar_module.base_kernel.lengthscale = initial_kernel_parameters['lengthscale']
        self.covar_module.outputscale = initial_kernel_parameters['outputscale']

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModel_ConstMean(gpytorch.models.ExactGP):

    def __init__(self,
                 train_x,
                 train_y,
                 initial_kernel_parameters,
                 likelihood
                 ):
        super(ExactGPModel_ConstMean, self).__init__(train_x, train_y, likelihood)

        dim = train_x.shape[1]

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = dim))

        self.covar_module.base_kernel.lengthscale = initial_kernel_parameters['lengthscale']
        self.covar_module.outputscale = initial_kernel_parameters['outputscale']

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPWithNNFeatureExtractor(gpytorch.models.ExactGP):

    def __init__(self,
                 train_x,
                 train_y,
                 initial_kernel_parameters,
                 likelihood
                 ):
        super(GPWithNNFeatureExtractor, self).__init__(train_x, train_y, likelihood)

        dim = train_x.shape[1]

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 2*dim))

        self.covar_module.base_kernel.lengthscale = initial_kernel_parameters['lengthscale']
        self.covar_module.outputscale = initial_kernel_parameters['outputscale']

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(1*dim, 2*dim),
            torch.nn.BatchNorm1d(2*dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*dim, 2*dim),
            torch.nn.BatchNorm1d(2*dim),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)