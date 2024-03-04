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


path_train = r"C:\Users\Sarim\OneDrive\Desktop\ViTs for DBRP\data\training_data"
path_test = r"C:\Users\Sari\OneDrive\Desktop\ViTs for DBRP\data\testing_data"


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

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.images = []
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            if os.path.isdir(cls_path):
                cls_images = [os.path.join(cls_path, img) for img in os.listdir(cls_path)]
                self.images.extend([(img, self.class_to_idx[cls]) for img in cls_images])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

data_transform = transforms.Compose([
    transforms.Resize((144, 144)),
  #[Augmentation if needed]
    transforms.ToTensor()
])


dataset = CustomDataset(root_dir=path_train, transform=data_transform)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, emb_size=256):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((patch_size **  2) * in_channels, emb_size)
        )

        #initialize linear layer weights randomly
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

    plt.show()

show_images(dataset)

    
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
print("Initial shape: ", sample_datapoint.shape) # 1 = batch dimention, 3 = color channels, dimensions = 144 by 144
embedding = PatchEmbedding()(sample_datapoint)
print("Patches shape: ", embedding.shape) # After applyign the patch embeedding, there are 324 patches, and each of the patches of a dimension of 128*128
                                        # We get the number 324, because number of patches = (image height/patch height)(image width/patch width)
                                        #AND our intial image shape was set to 144*144. 144 divided by our patch size of 8 equals to 18. 18 squared = 324


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
norm = PreNorm(256, Attention(dim=256, n_heads=4, dropout=0.1))
norm(torch.ones((1, 5, 256))).shape


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

train_split = int(0.8 * len(dataset))
train, test = random_split(dataset, [train_split, len(dataset) - train_split])

train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test, batch_size=32, shuffle=False)
correct_predictions = 0
total_samples = 0

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()
from torch.utils.data import SubsetRandomSampler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
  
train_subset_sampler = SubsetRandomSampler(range(100))
test_subset_sampler = SubsetRandomSampler(range(100))
subset_train_dataloader = DataLoader(train.dataset, batch_size=32, sampler=train_subset_sampler)
subset_test_dataloader = DataLoader(test.dataset, batch_size=32, sampler=test_subset_sampler)

lr = []
num_params = count_parameters(model)
print(f"Number of parameters in the model: {num_params}")
#========================================================================================================================
class EarlyStopping:
    def __init__(self, patience=4, verbose=False, delta=0, path=r'C:\Users\Sarim&Sahar\OneDrive\Desktop\vv\save_data.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.00000001, eps=1e-06, verbose=1)
earlystopping = EarlyStopping(patience=3, verbose=True)
#========================================================================================================================
num_epochs = 20

for epoch in range(num_epochs):  
    train_losses = []
    train_correct_predictions = 0
    train_total_samples = 0
    
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
    val_losses = []
    val_correct_predictions = 0
    val_total_samples = 0
    
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            

            _, predicted = torch.max(outputs, 1)
            val_correct_predictions += torch.sum(predicted == labels).item()
            val_total_samples += labels.size(0)

    val_accuracy = val_correct_predictions / val_total_samples
    
    print(f">>> Epoch {epoch+1} train loss: {np.mean(train_losses)} train accuracy: {train_accuracy}")
    print(f">>> Epoch {epoch+1} test loss: {np.mean(val_losses)} test accuracy: {val_accuracy}")


    val_loss_avg = np.mean(val_losses)
    earlystopping(val_loss_avg, model)
    scheduler.step(np.mean(val_losses))
    if earlystopping.early_stop:
        print("Early stopping")
        break

    # Update the learning rate based on the validation loss
    scheduler.step(np.mean(val_losses))
    print('epoch={}, learning rate={:.4f}'.format(epoch + 1, optimizer.state_dict()['param_groups'][0]['lr']))
    lr.append(optimizer.state_dict()['param_groups'][0]['lr'])


model.eval()
inputs, labels = next(iter(test_dataloader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = model(inputs)
print(outputs)
print("Predicted classes", outputs.argmax(-1))
print("Actual classes", labels)
