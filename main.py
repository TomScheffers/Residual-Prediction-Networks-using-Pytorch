from models.VGG import VGG
from utils.cifar100_conv_trainer import cifar100_trainer
from utils.plot import plot_results

device = 'cuda'
model1 = VGG(classifier_type="linear", lr=1e-3, device=device, hidden=200, output=100)
model2 = VGG(classifier_type="residual", lr=1e-3, device=device, hidden=120, groups=3, depth=5, output=100)

trainer1, trainer2 = cifar100_trainer(model1, device, batch_size=256), cifar100_trainer(model2, device, batch_size=256)

epoch_list, train_losses1, test_losses1, test_accs1 = trainer1.train_model(epochs=20)
epoch_list, train_losses2, test_losses2, test_accs2 = trainer2.train_model(epochs=20)
