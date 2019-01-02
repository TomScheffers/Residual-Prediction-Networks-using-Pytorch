import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.residual_prediction_net import ResidualPredictionNet
torch.manual_seed(42)

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, classifier_type, lr=1e-3, device="cpu", hidden=200, output=10, groups=5, depth=5, batch_norm=True):
        super(VGG, self).__init__()
        self.name = "VGG 16" + " with a linear classifier" if classifier_type == "linear" else " with a residual classifier"
        self.classifier_type = classifier_type

        # Define the VGG architecture
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #initialize the type of classifer
        if classifier_type == "linear":
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, hidden),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(hidden, hidden),
                nn.ReLU(True),
                nn.Linear(hidden, output)
            )
        else:
            self.classifier = ResidualPredictionNet(
                input_dim=512, hidden_dim=hidden, 
                output_dim=output, residual_depth=depth,
                groups=groups, batch_norm=batch_norm,
                dropout=True, device=device)
        
        #initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        # Putting it on the right device
        if device == "cuda":
            self.features = torch.nn.DataParallel(self.features)
            self.cuda()
            if classifier_type == "residual":
                self.classifier.cuda()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-3)

        self.total_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        print("Initiated a VGG net with ", self.total_params, " classifier parameters!\n")

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x