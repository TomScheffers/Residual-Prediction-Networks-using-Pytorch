import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, groups):
        super(ConvPredictor, self).__init__()
        self.feature_maps = input_dim
        self.groups = groups
        self.output_dim = output_dim
        self.conv = nn.Conv1d(in_channels=self.feature_maps, out_channels= self.groups * self.output_dim, kernel_size=1, groups=self.groups)

    def forward(self, x):
        x = x.unsqueeze(-1)
        #we stack the outputs of each group and sum them in the next step.
        outs = torch.stack(torch.split(self.conv(x), self.output_dim, dim=1))
        return outs.sum(0).reshape(-1, self.output_dim)

class ResidualPredictionNet(nn.Module):
    """
    This nn.module implements a PyTorch version of the Residual Prediction Network
    """
    def __init__(self, input_dim, hidden_dim, output_dim, residual_depth, groups, batch_norm=True, dropout=True, device="cpu"):
        """
        Args:
            input_dim: int for the amount of inputs
            hidden_dim: int for the amount of hidden neurons in the information core
            output_dim: int for the amount of outputs
            residual_depth: amount of layers in the RPN
            groups: the amount of groups in the prediction network. 1 group = regular conv layer
            batch_norm: boolean of whether or not to use batchnormalization before the ReLU layer
            dropout: boolean of whether or not to use dropout before the fully connected information layers
        """
        super(ResidualPredictionNet, self).__init__()
        self.name = "Residual Prediction Net"

        #Check if all input values are correct
        assert (input_dim >= 1 and hidden_dim >= 1 and output_dim >= 1 and residual_depth >= 1)
        assert (hidden_dim % groups == 0)
        assert (isinstance(input_dim, int) and isinstance(hidden_dim, int) and isinstance(output_dim, int) and isinstance(residual_depth, int))

        #Store all variables for the information network
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = residual_depth
        self.batch_norm = batch_norm
        self.dropout = True
        self.dropout_fn = nn.Dropout()
        self.device = device
        self.output_dim = output_dim

        #Define the information core of the Module
        self.core = []
        self.core += [nn.Linear(self.input_dim, self.hidden_dim)]
        self.core += [nn.Linear(self.hidden_dim, self.hidden_dim) for d in range(self.depth - 1)]
        if self.batch_norm:
            self.core_bn = [nn.BatchNorm1d(self.hidden_dim) for d in range(self.depth)]

        #Define the Predictors which predict from the core
        self.predictors = [ConvPredictor(self.hidden_dim, self.output_dim, groups) for d in range(self.depth)]

        #Put all modules into a list, so we can call module.parameters()
        extra = self.core_bn if self.batch_norm else []
        self.modules = nn.ModuleList(self.core + [v for v in self.predictors] + extra)

    def forward(self, x):
        """
        Args:
            x: input data to the module.
        Returns:
            out: returns the sum of the predictors
        """

        #first we calculate the activations of the core
        if self.dropout:
            x = self.dropout_fn(x)
        if self.batch_norm:
            core_outs = [F.relu(self.core_bn[0](self.core[0](x)))]
            for i in range(self.depth - 1):
                if self.dropout:
                    core_outs[-1] = self.dropout_fn(core_outs[-1])
                core_outs.append(F.relu(self.core_bn[i + 1](self.core[i + 1](core_outs[-1]))))
        else:
            core_outs = [F.relu(self.core[0](x))]
            for i in range(self.depth - 1):
                if self.dropout:
                    core_outs[-1] = self.dropout_fn(core_outs[-1])
                core_outs.append(F.relu(self.core[i + 1](core_outs[-1])))
        
        #Calculate the outputs of each Predictor
        predictor_outs = torch.stack([self.predictors[d](core_outs[d]) for d in range(self.depth)], dim=-1).sum(-1)
        return predictor_outs