import os
import sys
sys.path.append(os.path.abspath("DL_function.py"))
from DL_function import *

# Split the dataset
images_path = 'input/stanford-dogs-dataset/images/Images/'
results_path = 'input/model'
presplit = False
train_split = 0.5
val_split = 0.25
test_split = 0.25
batch_size = 128
threads = 0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

training_set_loader, testing_set_loader, validation_set_loader, dataset, training_set, testing_set, validation_set = \
                  load_transform_images(images_path, presplit, train_split, test_split, val_split, batch_size, threads, mean, std)

class_names = dataset.classes
class_names = [classes[10:] for classes in class_names]

# Load a pretrained model 
net_model = resnet.resnet152(pretrained=True)
net_name = 'resnet152'
unfrozen_layers = ['layer4', 'fc'] 
dropout_ratio = 0.9
net_model2 = load_network(net_model, net_name, dropout_ratio, class_names, unfrozen_layers)

# Train the model
learning_rate = 0.0001
epochs = 100
momentum = 0.9
weight_decay = 0
patience = 3
n_epochs_stop = 5

net_model, loss_acc, y_testing, preds = train_model(results_path, net_name, net_model, training_set_loader, validation_set_loader, 
                                                    learning_rate, epochs, momentum, weight_decay, patience, n_epochs_stop)

# Read the whole dataset
transform = transforms.Compose([
                               transforms.Resize((224,224)),
                               transforms.ToTensor(),
                               transforms.Normalize(torch.Tensor(mean),
                                                    torch.Tensor(std))])

images_path = 'input/stanford-dogs-dataset/images/Images/'
dataset = datasets.ImageFolder(root=images_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=threads, shuffle=False)

# Covert all images into vectors
predictions, trues = test_model(net_name, net_model, dataloader)
file_name = []
for i in range(len(dataset.imgs)):
    name = dataset.imgs[i][0].split('/')[-1]
    file_name.append(name)
true_class = trues.tolist()
predicted_class = predictions.argmax(1).tolist()
d = {'id': file_name, 'true_class':true_class, 'predicted_class':predicted_class}
pd.DataFrame(data=d).to_csv('info.csv')
pd.DataFrame(predictions).to_csv('vectors.csv')
