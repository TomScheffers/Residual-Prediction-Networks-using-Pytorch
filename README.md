# Residual prediction networks using Pytorch
The residual prediction network is an effort to improve the classification performance of neural networks.
This repo is used for testing the newly proposed hypothesis on the CIFAR10/100 datasets, however the class implementing the residual prediction network can be used as a drop-in replacement for fully connected layers. It can be found in the models folder.

The residual prediction network consists of two parts: the information- and prediction-network, as can be seen in the image below.

The information network aims to maintain valuable information for each of the predictors to use. It generally has a dimension between the input and output dimension in order to bottleneck while maintaining useful information for the prediction network. The dimensionality can also be decreased gradually when adding more depth.

The prediction network is designed to leverage both serial and parallel ensemble learning using residual connection and grouped convolutions respectively. The output of each group in each grouped convolution are summed up to determine the output of the network.

![residual prediction network](https://user-images.githubusercontent.com/28607837/50612769-81cf8b80-0edb-11e9-8072-16e6e8da5f20.png)
