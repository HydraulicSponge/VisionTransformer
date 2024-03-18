import numpy as np
import pandas as pd
import os, random, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, sampler
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import torchmetrics
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import torchvision.models as models
import torch.onnx
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()
path_train = '/kaggle/input/diabetic-retinopathy-resized-arranged'
classes = ['0', '1', '2', '3', '4']


for i in classes:
    class_path = os.path.join(path_train, i)
    num_images = len([file for file in os.listdir(class_path) if file.endswith(('jpg', 'jpeg', 'png'))])
    print(f"class: {i}, num of datapoints: {num_images}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
output_dir = '/kaggle/working/'


if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
   
def set_random_seed(seed: int) -> None:

    print(f"Setting seeds: {seed} ...... ")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=  True
   
def worker_init_fn(worker_id):    
                                                 
    np.random.seed(np.random.get_state()[1][0] + worker_id)

set_random_seed(123)

def make_weights_for_balanced_classes(labels):
    count = torch.bincount(torch.tensor(labels)).to(device)
    print('Count:', count.cpu().detach().numpy())
   
    weight = 1. / count.cpu().detach().numpy()
    print('Data sampling weight:', weight)
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)

    return samples_weight



path_val = path_train
path_test = path_train



batch_size = 32
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]         # Std of ImageNet dataset (used for normalization)



train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomAffine(degrees=20, translate=(0, 0), scale=(0.8, 1.2), shear=0),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.RandomVerticalFlip()], p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

dataset = ImageFolder(path_train, transform=train_transform)
dataset_test = ImageFolder(path_train, transform=test_transform)
targets = dataset.targets
               
train_idx, valid_idx= train_test_split(
np.arange(len(targets)),
test_size=0.2,
shuffle=True,
stratify=targets)

train_dataset = Subset(dataset, train_idx)#to_list()
val_dataset = Subset(dataset_test, valid_idx)
test_dataset = Subset(dataset_test, valid_idx)
train_dataset, val_dataset, test_dataset = train_dataset.dataset, val_dataset.dataset, test_dataset.dataset
import matplotlib.pyplot as plt
import random


label_mapping = {
    0: "healthy",
    1: "mild npdr",
    2: "moderate npdr",
    3: "severe npdr",
    4: "pdr"
}
def show_images(dataset, num_samples=20, cols=4):

    random_dataset = random.sample(list(range(len(dataset))), num_samples)
    plt.figure(figsize=(15, 15))
    for i, idx in enumerate(random_dataset):
        image, target = dataset[idx]
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(to_pil_image(image[0]))
        plt.colorbar()
        plt.title(label_mapping[target])
        plt.axis('on')
        plt.suptitle(f"Random {num_samples} images from the training dataset", fontsize=16, color="black")

    plt.show()

show_images(dataset)


                   
weights = make_weights_for_balanced_classes(train_dataset.targets)
weighted_sampler = sampler.WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                        num_workers=2, worker_init_fn=worker_init_fn , sampler=weighted_sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)




best_test_acc = 0
best_epoch = 0
num_classes = 5
train_losses = []  
val_losses = []  
train_accuracies = []
val_accuracies = []  
logs = ''



print("Model: ViT")
model = models.vit_b_16(weights='IMAGENET1K_V1').to(device)#weights='IMAGENET1K_V1'
#model.load_state_dict(torch.load('/kaggle/input/yoloweights/yolov5s-seg.pt'))

#model = torch.load('/kaggle/input/model-architecture/model_architecture (3).pth')##########################################load
#model.load_state_dict(torch.load('/kaggle/input/model-weights/model_weights (3).pth'))############################load



model.heads = nn.Sequential(
    nn.Linear(in_features=768, out_features=5, bias=True)


)

model = model.to(device)

class Temperature(nn.Module):
    def __init__(self, init_weight):
        super(Temperature, self).__init__()
        self.T = nn.Parameter(init_weight)

    def forward(self, x):
        return x / torch.exp(self.T)


count = torch.bincount(torch.tensor(train_dataset.targets)).to(device)
class_weight = len(train_dataset.targets) / count

print('Loss class weight:', class_weight)
criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
params = list(model.parameters())
optimizer = optim.SGD(params, lr=0.0001, weight_decay=1e-4)#, eps=1e-8)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, mode='min')

accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='weighted').to(device)
confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize='true').to(device)
class_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average=None).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f"Number of parameters in the model: {num_params}")

#TRAINING
num_epochs = 12
for epoch in range(num_epochs):
    t = tqdm(enumerate(train_loader, 0), total=len(train_loader),
                smoothing=0.9, position=0, leave=True,
                desc="Train: Epoch: "+str(epoch+1)+"/"+str(num_epochs))
    model.train()
    running_loss = 0.0
   
    for i, (inputs, labels) in t:
        inputs, labels = inputs.to(device).float(),labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels, weight=class_weight)
        criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        outputs = F.softmax(outputs, dim=-1)
        train_accuracy = accuracy(outputs, labels)
   
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
   
    train_accuracy = accuracy.compute()
    train_accuracies.append(float(train_accuracy))
    accuracy.reset()
   

    # Validation
    model.eval()
    val_correct = 0
    val_loss = 0.0
   
    with torch.no_grad():
        t = tqdm(enumerate(val_loader, 0), total=len(val_loader),
                smoothing=0.9, position=0, leave=True,
                desc="Val: Epoch: "+str(epoch+1)+"/"+str(num_epochs))
        for i, (inputs, labels) in t:
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, weight=class_weight)
            criterion(outputs, labels)
            val_loss += loss.item()
            outputs = F.softmax(outputs, dim=-1)
            val_accuracy = accuracy(outputs, labels)
            confmat.update(outputs, labels)
            val_class_accuracy = class_accuracy(outputs, labels)
           
   
    val_class_accuracy = class_accuracy.compute()  
 
    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = accuracy.compute()
    val_accuracies.append(float(val_accuracy))

    test_loss = val_loss
   

    test_accuracy = val_accuracy

    scheduler.step(val_loss)
    lr_log = f"LR: {optimizer.param_groups[0]['lr']}" # scheduler._last_lr
    print(lr_log)
    logs+=lr_log+'\n'
 
    train_results = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Training Accuracy: {train_accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}"
    print(train_results)
    logs+=train_results+'\n'
    torch.save(model.state_dict(), os.path.join(output_dir, f'last.pth')) #Save model checkpoint
   
    if (epoch+1)%5==0:
        torch.save(model.state_dict(), os.path.join(output_dir, f'epoch{epoch+1}.pth'))

     
        fig, ax = plt.subplots()
        confmat_vals = np.around(confmat.compute().cpu().detach().numpy(), 3)
        im = ax.imshow(confmat_vals)

        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('True class')

     
        for i in range(num_classes):
            for j in range(num_classes):
                text = ax.text(j, i, confmat_vals[i, j],ha="center", va="center", color="black", fontsize=12)

        ax.set_title(f"Confusion Matrix on Test for epoch {epoch+1}")
        fig.savefig(os.path.join(output_dir, f"conf_mat_epoch{epoch+1}.png"))
        plt.close()
     
   
    if best_test_acc <= test_accuracy and epoch!=0:
        best_epoch = epoch+1
        log = f"Improve accuracy from {best_test_acc} to {test_accuracy}"
        print(log)
        logs+=log+"\n"
        best_test_acc = test_accuracy
        torch.save(model.state_dict(), os.path.join(output_dir, f'best.pth'))
       
   
        fig, ax = plt.subplots()
        confmat_vals = np.around(confmat.compute().cpu().detach().numpy(), 3)
        im = ax.imshow(confmat_vals)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('True class')

        # Loop over data dimensions and create text annotations.
        for i in range(num_classes):
            for j in range(num_classes):
                text = ax.text(j, i, confmat_vals[i, j],ha="center", va="center", color="black", fontsize=12)

        ax.set_title("Confusion Matrix on Test for best model")
        fig.savefig(os.path.join(output_dir, "conf_mat_best.png"))
        plt.close()
   
 
    accuracy.reset(); class_accuracy.reset(); confmat.reset()
   

#Save the printed outputs to a log.txt file
with open(os.path.join(output_dir, 'log.txt'), 'w') as log_file:
    log_file.write(logs)
    log_file.write(f'Best val accuracy: {best_test_acc} in epoch {best_epoch}')
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Validation Accuracy')

plt.savefig(os.path.join(output_dir, 'loss_accuracy_graph.png'))
plt.close()



torch.save(model, os.path.join('model_architecture.pth'))#############################################save
torch.save(model.state_dict(), os.path.join(output_dir, f'model_weights.pth'))########################save


#TESTING
model.eval()

test_loss = 0
test_acc = 0
with torch.no_grad():
    for x, y in test_loader:
        x,y = x.to(device), y.to(device)
        pred = model(x)
 
        loss = criterion(pred, y)

        test_loss += loss.item()

        test_acc += accuracy(pred, y)

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
print(f'Test Accuracy: {test_acc}, Test Loss: {test_loss}')    



#TESTING 2
model.eval()
with torch.no_grad():
    inputs, labels = next(iter(test_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    print(outputs)
    print("Predicted classes", outputs.argmax(-1))
    print("Actual classes", labels)

print("DR classifier model completed")
