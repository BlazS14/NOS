import cv2
import tkinter as tk
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import os
import random
from torch.utils.data import Dataset, DataLoader
from math import log10, sqrt
from skimage.metrics import structural_similarity as SSIM
import torchinfo


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1,64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.batchnorm = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3136, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(128,1, kernel_size=7, stride=1, padding=3)
        self.batchnorm = nn.BatchNorm2d(128)
        self.linear = nn.Linear(100, 6272)
        self.sigmoid = nn.Sigmoid()
        self.convtranspose1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = torch.reshape(x,(-1,128, 7, 7))
        x = self.convtranspose1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.convtranspose1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x
    
class ImageDataSet(Dataset):
    def __init__(self):
        mnist_ds = torchvision.datasets.MNIST(
        root="datasets", train=True, transform=torchvision.transforms.ToTensor(),
        download=True)
        print(f"Širina slik: {mnist_ds[0][0].shape[2]}")
        print(f"Višina slik: {mnist_ds[0][0].shape[1]}")
        print(f"Število kanalov: {mnist_ds[0][0].shape[0]}")
        print(f"Število slik: {len(mnist_ds)}")
        print(f"Podatkovna zbirka: MNIST (http://yann.lecun.com/exdb/mnist/)")

        n_rows = 5
        n_cols = 5
        _, axes = plt.subplots(n_rows, n_cols)
        for r in range(n_rows):
            for c in range(n_cols):
                img, _ = mnist_ds[random.randint(0, len(mnist_ds) - 1)]
                axes[r, c].imshow(img.permute(1, 2, 0), cmap="gray")
                axes[r, c].axis("off")

        plt.tight_layout()
        plt.show()
        
        self.images = mnist_ds
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img, _ = self.images[index]
        img = img/255
        return img
    
def create_identity(num):
    identity_array = np.empty((0,100), dtype=np.float32)
    for i in range(num):
        identity = np.random.standard_normal(100)
        identity = (identity + abs(np.min(identity)))
        identity = identity / np.max(identity)
        identity_array = np.append(identity_array, [identity], axis=0)       
    identity_array = torch.from_numpy(identity_array.astype(np.float32))
    return identity_array

def train(discriminator_model, discriminator_optimizer, discriminator_loss_fn, generator_model, generator_optimizer, generator_loss_fn, epochs, batch_size,dataset,static_identity):
    discriminator_loss_sum = 0
    generator_loss_sum = 0
    for epoch in range(1,epochs+1):
        discriminator_optimizer.zero_grad()
        data = next(iter(dataset))
        discriminator_output = discriminator_model(data)
        discriminator_loss = discriminator_loss_fn(discriminator_output, torch.ones(batch_size,1))
        discriminator_loss.backward()
        generator_optimizer.zero_grad()
        generator_output = generator_model(create_identity(batch_size))
        discriminator_output = discriminator_model(generator_output)
        generator_loss = generator_loss_fn(discriminator_output, torch.ones(batch_size,1))
        generator_loss.backward()
        generator_optimizer.step()
        discriminator_loss = discriminator_loss_fn(discriminator_output, torch.zeros(batch_size,1))
        discriminator_loss.backward()
        discriminator_optimizer.step()
        
        img = generator_model(static_identity)
        img = img * 255
        img = img.detach().numpy()[0,0,:,:]
        img = img.astype(np.uint8)
        
        '''discriminator_optimizer.zero_grad()
        generator_optimizer.zero_grad()
        generator_model.zero_grad()
        generator_output = generator_model(create_identity(batch_size))
        discriminator_output = discriminator_model(generator_output)
        generator_loss = generator_loss_fn(discriminator_output, torch.ones(batch_size,1))
        generator_loss.backward()
        generator_optimizer.step()'''
        
        
        
        generator_loss_sum += generator_loss.item()
        discriminator_loss_sum += discriminator_loss.item()
    
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Generator loss: {generator_loss_sum/10}, Discriminator loss: {discriminator_loss_sum/10}")
            generator_loss_sum = 0
            discriminator_loss_sum = 0
            cv2.imwrite("img"+str(epoch)+".png",img)
    
    
disc_model = Discriminator()
disc_optim = torch.optim.Adam(disc_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_loss_fn = nn.BCELoss()

gen_model = Generator()
gen_optim = torch.optim.Adam(gen_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
gen_loss_fn = nn.BCELoss()

data_set = ImageDataSet()
data_gen = DataLoader(data_set, batch_size=128,shuffle=True)


train(disc_model, disc_optim, disc_loss_fn, gen_model, gen_optim, gen_loss_fn, 
      1000, 128, data_gen,create_identity(1))

torch.save(disc_model, "model_Discriminator.pt")
torch.save(gen_model, "model_Generator.pt")