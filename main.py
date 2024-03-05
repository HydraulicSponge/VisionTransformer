from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor
from einops import repeat
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
from torchvision.transforms import Resize, ToTensor


path_train = r"C:\Users\Sarim&Sahar\OneDrive\Desktop\ViTs for DBRP\data\training_data"
path_test = r"C:\Users\Sarim&Sahar\OneDrive\Desktop\ViTs for DBRP\data\testing_data"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
        return image, target

classes = ['healthy', 'mild npdr', 'moderate npdr', 'severe npdr', 'pdr']


for i in classes:
    class_path = os.path.join(path_train, i)
    num_images = len([file for file in os.listdir(class_path) if file.endswith(('jpg', 'jpeg', 'png'))])
    print(f"class: {i}, num of datapoints: {num_images}")


from torchvision import transforms

"""
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Shuffle image paths
        self.image_paths = [] 
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            self.image_paths.extend(glob.glob(os.path.join(cls_path, '*.*')))
        random.shuffle(self.image_paths) 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
    
    # Get class name from path
        class_name = img_path.split('/')[-2]  

    # Get class index 
        label = self.class_to_idx[class_name]

        return image, label
"""
from torchvision.datasets import ImageFolder
data_transform = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor()
    ])


dataset = ImageFolder(root=path_train, transform=data_transform)

import matplotlib.pyplot as plt
import random

num_rows = 5
num_cols = 5


fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
to_pil = transforms.ToPILImage() 

for i in range(num_rows):
    for j in range(num_cols):

        image_index = random.randrange(len(dataset))

        axs[i, j].imshow(dataset[image_index][0].permute((1, 2, 0)))


        axs[i, j].set_title(dataset.classes[dataset[image_index][1]], color="black")
        image, label = dataset[image_index]
        pil_image = to_pil(image)

        axs[i, j].axis(False)
        axs[i, j].imshow(pil_image)


fig.suptitle(f"Random {num_rows * num_cols} images from the training dataset", fontsize=16, color="blue")

fig.set_facecolor(color='white')

plt.show()

"""
def show_images(dataset, num_samples=20, cols=4):
    # Get a random subset of indices
    random_dataset = random.sample(list(range(len(dataset))), num_samples)
    plt.figure(figsize=(15, 15))
    for i, idx in enumerate(random_dataset):
        image, target = dataset[idx]
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(to_pil_image(image[0]))
        plt.colorbar()
        plt.title(label_mapping[target])
        plt.axis('on')

    plt.show()

show_images(dataset)"""

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, emb_size=256):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((patch_size **  2) * in_channels, emb_size)
        )

        # Initialize the linear layer weights randomly
        nn.init.xavier_uniform_(self.projection[1].weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x


label_mapping = {
    0: "healthy",
    1: "mild npdr",
    2: "moderate npdr",
    3: "severe npdr",
    4: "pdr"
}


    
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=0.1)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(q, k, v)
        attn_output = self.norm(self.dropout(attn_output) + x)
        return attn_output

Attention(dim=256, n_heads=4, dropout=0.1)(torch.ones((1, 5, 256))).shape



sample_datapoint = torch.unsqueeze(dataset[0][0], 0)
print("Initial shape: ", sample_datapoint.shape) 
embedding = PatchEmbedding()(sample_datapoint)
print("Patches shape: ", embedding.shape)




class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
norm = PreNorm(256, Attention(dim=256, n_heads=4, dropout=0.1))
norm(torch.ones((1, 5, 256))).shape

#==

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))) + x))
        return x

ff = FeedForward(dim=256, hidden_dim=512)
ff(torch.ones((1, 5, 256))).shape

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

residual_att = ResidualAdd(Attention(dim=256, n_heads=4, dropout=0.1))
residual_att(torch.ones((1, 5, 256))).shape
import math
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()
class ModelArgs:
    dim: int = 256         
    hidden_dim: int = 512  
    n_heads: int = 8        
    n_layers: int = 6       
    patch_size: int = 8     
    n_channels: int = 3     
    n_patches: int = 1024  
    n_classes: int = 5   
    dropout: float = 0.2
    in_channels: int=3   
class ViT(nn.Module):
    def __init__(self, args):
        super(ViT, self).__init__()
        self.patch_size = args.patch_size
        self.in_channels = args.in_channels
        self.hidden_dim = args.hidden_dim
        self.dim = args.dim
        self.n_classes = args.n_classes
        self.dropout = args.dropout
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.n_patches = args.n_patches

        self.patch_embed = PatchEmbedding(in_channels=self.in_channels, patch_size=self.patch_size, emb_size=self.dim)
        self.pos_encoding = PositionalEncoding(emb_size=self.dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.n_heads, dim_feedforward=self.hidden_dim, dropout=self.dropout), num_layers=self.n_layers)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.n_classes)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == w, 'Input tensor shape must be square'

        x = self.patch_embed(x)
        x = x.permute(1, 0, 2)  
        cls_token = self.cls_token.expand(-1, b, -1)
        x = torch.cat((cls_token, x), dim=0)
        x = self.pos_encoding(x)

        x = self.transformer(x)

        cls_token_final = x[0]


        x = self.mlp_head(cls_token_final)

        return x


device = "cpu"
model = ViT(args=ModelArgs()).to(device)
model(torch.ones((1, 3, 144, 144)))

test_dataset = ImageFolder(root=path_test, transform=data_transform)

train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
correct_predictions = 0
total_samples = 0

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)

criterion = nn.CrossEntropyLoss()
from torch.utils.data import SubsetRandomSampler


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
train_subset_sampler = SubsetRandomSampler(range(100))
test_subset_sampler = SubsetRandomSampler(range(100))

#subset_train_dataloader = DataLoader(dataset, batch_size=32, sampler=train_subset_sampler)
#subset_test_dataloader = DataLoader(test_dataset, batch_size=32, sampler=test_subset_sampler)
lr = []
num_params = count_parameters(model)
print(f"Number of parameters in the model: {num_params}")
from pytorchtools import EarlyStopping
scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.00000001, eps=1e-06, verbose=1)
earlystopping = EarlyStopping(patience=3, verbose=True)
num_epochs = 1
for epoch in range(num_epochs):  
    train_losses = []
    train_correct_predictions = 0
    train_total_samples = 0
    
    # Training phase
    model.train()
    for step, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        

        _, predicted = torch.max(outputs, 1)
        train_correct_predictions += torch.sum(predicted == labels).item()
        train_total_samples += labels.size(0)

    train_accuracy = train_correct_predictions / train_total_samples

    model.eval()
    test_losses = []
    test_correct_predictions = 0
    test_total_samples = 0
    
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
            
            #_, predicted = torch.max(outputs, 1)
            test_correct_predictions += torch.sum(predicted == labels).item()
            test_total_samples += labels.size(0)

    test_accuracy = test_correct_predictions / test_total_samples
    
    print(f">>> Epoch {epoch+1} train loss: {np.mean(train_losses)} train accuracy: {train_accuracy}")
    print(f">>> Epoch {epoch+1} test loss: {np.mean(test_losses)} test accuracy: {test_accuracy}")


    test_loss_avg = np.mean(test_losses)
    earlystopping(test_loss_avg, model)
    scheduler.step(np.mean(test_losses))
    if earlystopping.early_stop:
        print("Early stopping")
        break


    scheduler.step(np.mean(test_losses))
    print('epoch={}, learning rate={:.4f}'.format(epoch + 1, optimizer.state_dict()['param_groups'][0]['lr']))
    lr.append(optimizer.state_dict()['param_groups'][0]['lr'])


model.eval()
inputs, labels = next(iter(test_dataloader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = model(inputs)
print(outputs)
print("Predicted classes", outputs.argmax(-1))
print("Actual classes", labels)
