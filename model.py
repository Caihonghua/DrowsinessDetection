import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS= 32
TS=(24,24)
train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('data/valid',shuffle=True, batch_size=BS,target_size=TS)
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
print(SPE,VS)


# img,labels= next(train_batch)
# print(img.shape)

model = Sequential([
    #卷基层，激活韩式relu，输入格式是24*24*1，卷积核大小3*3
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    #池化层大小1*1
    MaxPooling2D(pool_size=(1,1)),
    #池化层，激活函数是relu
    Conv2D(32,(3,3),activation='relu'),
    #池化层，大小1*1
    MaxPooling2D(pool_size=(1,1)),
    #再重复一次卷积池化，但是下一次的大小就是64，而不是32
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

    # 卷积过滤器大小3x3
    # 选择最适合的特征给via pooling
    
    # 为了更好地改善收敛的情况，随机关闭/打开节点
    Dropout(0.25),
    # 如果维度太多的时候，我们需要采用此方法去化简我们的输出
    # 详细情况你可以去看一下tensorflow的flatten
    Flatten(),
    # 把所有相关的数据全链接起来
    Dense(128, activation='relu'),
    # 为了收敛，dropout多一次
    Dropout(0.5),
    # 输出一个softmax
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)

model.save('models/cnnCat2.h5', overwrite=True)