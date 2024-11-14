from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import color
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image # This will be used to read/modify images (can be done via OpenCV too)
from numpy import *
import pickle
import random

# define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

Categories=['C','D', 'L','R','S','T','B'] 
flat_data_arr=[] #input array 
target_arr=[] #output array 
datadir='nowy_out5/' 
#path which contains all the categories of images 
data= []
labels = []

data_r = []
data_c = []
data_d = []
data_l = []
data_s = []
data_t = []
data_b = []

label_r = []
label_c = []
label_d = []
label_t = []
label_s = []
label_b = []
label_l = []

for i in Categories: 
    if(i == 'R'):    
        path=os.path.join(datadir,i) 
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista:
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_r.append(fd)
            label_r.append(i)
    if(i == 'C'):    
        path=os.path.join(datadir,i) 
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista:
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_c.append(fd)
            label_c.append(i)
    if(i == 'D'):    
        path=os.path.join(datadir,i)
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista: 
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_d.append(fd)
            label_d.append(i)
    if(i == 'L'):    
        path=os.path.join(datadir,i) 
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista:
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_l.append(fd)
            label_l.append(i)
    if(i == 'S'):    
        path=os.path.join(datadir,i) 
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista:
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_s.append(fd)
            label_s.append(i)
    if(i == 'T'):    
        path=os.path.join(datadir,i) 
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista:
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_t.append(fd)
            label_t.append(i)
    if(i == 'B'):    
        path=os.path.join(datadir,i) 
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista:
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_b.append(fd)
            label_b.append(i)


datadir = 'nowy_out3/'
for i in Categories: 
    if(i == 'R'):    
        path=os.path.join(datadir,i) 
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista:
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_r.append(fd)
            label_r.append(i)
    if(i == 'C'):    
        path=os.path.join(datadir,i) 
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista:
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_c.append(fd)
            label_c.append(i)
    if(i == 'D'):    
        path=os.path.join(datadir,i)
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista: 
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_d.append(fd)
            label_d.append(i)
    if(i == 'L'):    
        path=os.path.join(datadir,i) 
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista:
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_l.append(fd)
            label_l.append(i)
    if(i == 'S'):    
        path=os.path.join(datadir,i) 
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista:
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_s.append(fd)
            label_s.append(i)
    if(i == 'T'):    
        path=os.path.join(datadir,i) 
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista:
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_t.append(fd)
            label_t.append(i)
    if(i == 'B'):    
        path=os.path.join(datadir,i) 
        lista = os.listdir(path)
        random.shuffle(lista)
        for im in lista:
            img = Image.open(os.path.join(path,im))
            img = img.resize((64,64))
            gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
            # data.append(fd)
            # labels.append(i)
            data_b.append(fd)
            label_b.append(i)

k = 0.8
div_r = int(k * len(data_r))
div_c = int(k * len(data_c))
div_l = int(k * len(data_l))
div_d = int(k * len(data_d))
div_s = int(k * len(data_s))
div_t = int(k * len(data_t))
div_b = int(k * len(data_b))

trainData = data_r[:div_r]+data_c[:div_c]+data_d[:div_d]+data_l[:div_l]+data_t[:div_t]+data_s[:div_s]+data_b[:div_b]
trainLabels = label_r[:div_r]+label_c[:div_c]+label_d[:div_d]+label_l[:div_l]+label_t[:div_t]+label_s[:div_s]+label_b[:div_b]

testData = data_r[div_r:]+data_c[div_c:]+data_d[div_d:]+data_l[div_l:]+data_t[div_t:]+data_s[div_s:]+data_b[div_b:]
testLabels = label_r[div_r:]+label_c[div_c:]+label_d[div_d:]+label_l[div_l:]+label_t[div_t:]+label_s[div_s:]+label_b[div_b:]


#%%
# Partitioning the data into training and testing splits, using 80%
# of the data for training and the remaining 20% for testing
print(" Constructing training/testing split...")
# (trainData, testData, trainLabels, testLabels) = train_test_split(
# 	np.array(data), labels, test_size=0, random_state=42)
#%% Train the linear SVM
print(" Training Linear SVM classifier...")
model = LinearSVC()
model.fit(trainData, trainLabels)
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))


#%% Evaluate the classifier
# print(" Evaluating classifier on test data ...")
# predictions = model.predict(testData)
# print(classification_report(testLabels, predictions))


# save the model
with open('v12.pkl','wb') as f:
    pickle.dump(model,f)
    f.close()
