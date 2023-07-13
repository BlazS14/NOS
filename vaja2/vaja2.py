import cv2
import tkinter as tk
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import os
import random
from torch.utils.data import Dataset, DataLoader
from math import log10, sqrt
from skimage.metrics import structural_similarity as SSIM
import torchinfo

def PSNR(original, compressed):
    
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def calculate_frame_offsets(dirpath):
    offests = np.empty((0,3), dtype=np.int32)
    dirlist = os.listdir(dirpath)
    for n in range(len(dirlist)):
        img = cv2.imread(dirpath+dirlist[n])
        for i in range(0,img.shape[0]-32,14):
            for j in range(0,img.shape[1]-32,14):
                offests = np.append(offests,[[n,i,j]],axis=0)
    return offests

def get_images(dirpath):
    files = []
    for file in os.listdir(dirpath):
        files.append(dirpath+file)
    return files
        
                
class ImageDataSet(Dataset):
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.offsets = calculate_frame_offsets(dirpath)
        
    def __len__(self):
        return len(self.offsets)
    
    def __getitem__(self, index):
        dirlist = os.listdir(self.dirpath)
        img_orig = cv2.imread(self.dirpath+dirlist[self.offsets[index][0]])
        scale = random.randint(2,4)
        img = cv2.resize(img_orig, (int(img_orig.shape[1]/scale), int(img_orig.shape[0]/scale)))
        img = cv2.resize(img, (int(img_orig.shape[1]), int(img_orig.shape[0])))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2YCrCb)
        img = img[self.offsets[index][1]:self.offsets[index][1]+33,self.offsets[index][2]:self.offsets[index][2]+33,:]
        img_orig = img_orig[self.offsets[index][1]:self.offsets[index][1]+33,self.offsets[index][2]:self.offsets[index][2]+33,:]
        img_orig = img_orig/255
        img = img/255
        return np.array(img).astype(np.float32),np.array(img_orig).astype(np.float32)         
    
class ImageDataTestSet(Dataset):
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.files = get_images(dirpath)
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  
        img = img/255
        return np.array(img).astype(np.float32)      
                
#downscale and upscale image from path
def preprocess(path,scale):
    img_orig = cv2.imread(path)
    img = cv2.resize(img_orig, (int(img_orig.shape[1]/scale), int(img_orig.shape[0]/scale)))
    img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    
    return torch.from_numpy(img),torch.from_numpy(img_orig)

#slice image into patches of size 33x33 with stride 14
def slice_image(img,img_orig,framecounter):
    flag = False
    counter = 0
    patches = np.empty((0,33,33,3), dtype=np.float32)
    patches_orig = np.empty((0,33,33,3), dtype=np.float32)
    for i in range(0,img.shape[0]-32,14):
        for j in range(0,img.shape[1]-32,14):
            counter += 1
            if counter < framecounter:
                break
            framecounter += 1
            flag = True
            patches = np.append(patches,[img[i:i+33,j:j+33,:]],axis=0)
            patches_orig = np.append(patches_orig,[img_orig[i:i+33,j:j+33,:]],axis=0)
    return patches,patches_orig,flag
         
def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
        n,c,w,h = tensor.shape

        if allkernels: tensor = tensor.view(n*c, -1, w, h)
        elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

        rows = np.min((tensor.shape[0] // nrow + 1, 64))    
        grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
        plt.figure( figsize=(nrow,rows) )
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
                
class SuperRes(nn.Module):
    def __init__(self):
        super(SuperRes, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = x[:,:,:,0]
        x1 = self.conv1(x1)
        x1 = self.relu(x1)
        x1 = self.relu(self.conv2(x1))
        x1 = self.conv3(x1)
        return x1
    
def train(num_images,model, optimizer, loss_fn, epochs, print_step):
    train_set = ImageDataSet('C:\\Users\\GTAbl\\Desktop\\NOS\\vaja2\T91\\train\\')
    train_generator = DataLoader(train_set, batch_size=num_images,shuffle=True)
    loss_sum = 0
    for epoch in range(epochs):
        for img,img_orig in train_generator:
            for i in range(len(img)):
                optimizer.zero_grad()
                output = model(img[i:i+1,:,:,0:1])
                loss = loss_fn(output[:,:,:], img_orig[i:i+1,:,:,0])
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
               
            break
        if epoch % print_step == 0:
            print('Epoch %d loss: %.6f' % (epoch, loss.item()))
            loss_sum = 0 
    torch.save(model, "model_SuperRes.pt")
    
def test(num_images,dir_path):
    model = torch.load("model_SuperRes.pt")
    torchinfo.summary(model, (1,1, 33, 33))
    model.eval()
    #dir_path = 'C:\\Users\\GTAbl\\Desktop\\NOS\\vaja2\set14\\test\\'
    test_dir = os.listdir(dir_path)
    counter = 0
    
    psnr_hr = 0
    psnr_lr = 0
    ssim_hr = 0
    ssim_lr = 0
    
    for img_path in test_dir:
        counter += 1        
        scale = random.randint(2,4)
        img = cv2.imread(dir_path+img_path)
        
        orig_img = img.copy()
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2YCrCb)
        
        img = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)))
        img = cv2.resize(img, (int(orig_img.shape[1]), int(orig_img.shape[0])))
        cv2.imwrite(dir_path+'LR_'+img_path,img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        psnr_lr += PSNR(orig_img[:,:,0],img[:,:,0])
        ssim_lr += SSIM(orig_img[:,:,0],img[:,:,0])
        
        img = img/255
        img = torch.from_numpy(np.expand_dims(np.array(img).astype(np.float32),axis = 0))
        output = model(img[:,:,:,0:1])
        img[0,:,:,0] = output[0,:,:]
        img = img * 255
        img = img.detach().numpy()[0,:,:,:]
        img = img.astype(np.uint8)
        
        psnr_hr += PSNR(orig_img[:,:,0],img[:,:,0])
        ssim_hr += SSIM(orig_img[:,:,0],img[:,:,0])
        
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(dir_path+'SR_'+img_path,img)
        if counter == num_images:
            break
        
    filter = model.conv1.weight.data.clone()
    visTensor(filter, ch=0, allkernels=False)
    plt.axis('off')
    plt.ioff()
    plt.show()
        
    psnr_lr /= counter
    psnr_hr /= counter 
    ssim_lr /= counter
    ssim_hr /= counter
    
    

    print('Testing finished!')
    print('PSNR LR: %.6f' % (psnr_lr))
    print('PSNR HR: %.6f' % (psnr_hr))
    print('SSIM LR: %.6f' % (ssim_lr))
    print('SSIM HR: %.6f' % (ssim_hr))


def get_avg_sizes(dir_path):
    test_dir = os.listdir(dir_path)
    arr_heights = []
    arr_widths = []
    for img_path in test_dir:
        img = cv2.imread(dir_path+img_path)
        arr_heights.append(img.shape[0])
        arr_widths.append(img.shape[1])
    avg_height = sum(arr_heights) / len(arr_heights)
    avg_width = sum(arr_widths) / len(arr_widths)
    height_interval = [np.min(arr_heights),np.max(arr_heights)]
    width_interval = [np.min(arr_widths),np.max(arr_widths)]
    number_of_images = len(arr_heights)
    print('Average height: %.6f' % (avg_height))
    print('Average width: %.6f' % (avg_width))
    print('Height interval: ',height_interval)
    print('Width interval: ',width_interval)
    print('Number of images: ',number_of_images)

'''lol = cv2.imread('C:\\Users\\GTAbl\\Desktop\\NOS\\vaja2\T91\\train\\tt27.png')
lol = cv2.cvtColor(lol, cv2.COLOR_BGR2YCrCb)
lol = cv2.cvtColor(lol, cv2.COLOR_YCrCb2BGR)

cv2.imwrite('C:\\Users\\GTAbl\\Desktop\\NOS\\vaja2\T91\\train\\tt27LOL.png',lol)
'''
get_avg_sizes('C:\\Users\\GTAbl\\Desktop\\NOS\\vaja2\set14\\test\\')
get_avg_sizes('C:\\Users\\GTAbl\\Desktop\\NOS\\vaja2\set5\\test\\')
get_avg_sizes('C:\\Users\\GTAbl\\Desktop\\NOS\\vaja2\T91\\train\\')


model = SuperRes()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()
epochs = 5000
num_images = 32

#train(num_images,model, optimizer, loss_fn, epochs, 1000)
test(-1,'C:\\Users\\GTAbl\\Desktop\\NOS\\vaja2\set5\\test\\')
