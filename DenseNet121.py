#!/usr/bin/env python
# coding: utf-8

# In[9]:


import warnings  
warnings.filterwarnings('ignore')
import itertools
import os
import numpy as np
import pandas as pd
from scipy import interp
from datetime import datetime
from inspect import signature
import matplotlib.pyplot as plt
from collections import Counter
from glob import *
from tensorflow import keras
# from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras  import Input
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import plot_model
import cv2
from sklearn.metrics import *
print(keras.__version__)


# In[10]:



dir_name = 'MODEL/DenseNet121_model_'

try:
    os.mkdir('MODEL')
    print('Dir. Created')
except:
    print('AD')
try:
    os.mkdir(dir_name)
    print('Dir. Created')
except:
    print('AD')


# In[13]:


imagepath = os.getcwd()
training_data_dir = imagepath+'\\data\\Train\\' 
validation_data_dir = imagepath+'\\data\\Val\\'
test_data_dir = imagepath+'\\data\\Test\\' 
 
IMAGE_WIDTH, IMAGE_HEIGHT = 224,224               

epochs =  5                           
BATCH_SIZE_Train = 8                                
BATCH_SIZE_Val = 8                              
BATCH_SIZE_Test = 1                        

monitors = 'val_accuracy'                       
input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)       

 

noClass = 15                       


# In[14]:


training_data_generator = ImageDataGenerator(rescale=1./255,rotation_range=90,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='wrap')
 
training_generator = training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE_Train,
    class_mode="categorical",
    color_mode="rgb",                         # if colour image set rgb 
    shuffle=True)
  
validation_data_generator = ImageDataGenerator(rescale=1./255,rotation_range=90,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='wrap')

validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE_Val,
    class_mode="categorical",
    color_mode="rgb",                         # if colour image set rgb
    shuffle=True)


test_data_generator = ImageDataGenerator(rescale=1./255,rotation_range=90,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='wrap')

test_generator = test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE_Test,
    class_mode="categorical", 
    color_mode="rgb",                         # if colour image set rgb
    shuffle=False)   


# In[15]:



early_stopping = EarlyStopping(monitor= monitors,
                              patience=15,
                              verbose=1,
                              mode='max')

reduce_lr = ReduceLROnPlateau(monitor= monitors,
                              factor=0.1,
                              patience=5,
                              verbose=1,
                              mode='max',
                              min_lr=0.00000001)

filepath=f"./{dir_name}/model.hdf5" 
modelSave = ModelCheckpoint(filepath,
                       monitor= monitors,
                       verbose=1,
                       save_best_only=True,
                       save_weights_only=True,
                       mode='max')
 


print(training_generator.class_indices,training_generator.n)
print(validation_generator.class_indices,validation_generator.n)
print(test_generator.class_indices,test_generator.n)


# class_list= ['class: E','class: L','class: P'] 


# In[16]:


# vbhnjmk


# In[17]:






inputImage = Input(shape=input_shape)



input_shape = (224, 224, 3)

# model = Sequential([
#     Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
#            activation='relu'),
#     Conv2D(64, (3, 3), activation='relu', padding='same'),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    
    
#     Conv2D(128, (3, 3), activation='relu', padding='same'),
#     Conv2D(128, (3, 3), activation='relu', padding='same',),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    
    
#     Conv2D(256, (3, 3), activation='relu', padding='same',),
#     Conv2D(256, (3, 3), activation='relu', padding='same',),
#     Conv2D(256, (3, 3), activation='relu', padding='same',),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    
    
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    
    
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     Conv2D(512, (3, 3), activation='relu', padding='same',),
#     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    
    
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(128, activation='relu'),
#     Dense(noClass, activation='softmax')
# ])

DenseNet  = DenseNet121(weights=None, include_top=False,
                    input_tensor=inputImage, input_shape=input_shape)

X = DenseNet.output 
x = GlobalAveragePooling2D()(X)
x = Dense(512, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(noClass, activation='softmax')(x)

model = Model(inputs=inputImage, outputs=output)
model.summary()


# %%time

opt=Adamax(lr=0.0001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.summary()


# In[18]:




try:
 
    print ('Weights are Loading............................')
    model.load_weights(filepath)
    print ('Weights loading Done!!!!!......................')

except:
    print('No pretrained weight found')

#history = model.fit_generator(training_generator,
 #                   steps_per_epoch=len(training_generator.classes)//training_generator.batch_size,
  #                  epochs=epochs,
   #                 verbose=1,
    #                validation_data=validation_generator,
     #               validation_steps=len(validation_generator.classes)//validation_generator.batch_size,
      #               
       #             callbacks= [early_stopping,reduce_lr,modelSave])


# history = model.fit_generator(training_generator,
#                     steps_per_epoch=10,
#                     epochs=5,
#                     verbose=1,
#                     validation_data=validation_generator,
#                     validation_steps=10,
                     
#                     callbacks= [early_stopping,reduce_lr,modelSave])


# In[19]:


history = model.fit_generator(training_generator,
                    steps_per_epoch=len(training_generator.classes)//training_generator.batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator.classes)//validation_generator.batch_size,
                     
                    callbacks= [early_stopping,reduce_lr,modelSave])


# history = model.fit_generator(training_generator,
#                     steps_per_epoch=10,
#                     epochs=5,
#                     verbose=1,
#                     validation_data=validation_generator,
#                     validation_steps=10,
                     
#                     callbacks= [early_stopping,reduce_lr,modelSave])


# In[ ]:





# In[20]:



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(f'./{dir_name}/Acc vs Epoch .png', bbox_inches='tight',dpi=300)
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(f'./{dir_name}/Loss vs epoch.png', bbox_inches='tight')
plt.show()


# In[21]:


print(test_generator.class_indices,test_generator.n)


# In[22]:


model.load_weights(filepath)
Y_p = model.predict_generator(test_generator,
                               steps = len(test_generator.classes) //test_generator.batch_size,
                               verbose=1)
Y_p_class = np.argmax(Y_p, axis=1)
Y_p_class_categorical = keras.utils.to_categorical(Y_p_class,num_classes=noClass, dtype='float32')


# In[23]:


np.unique(Y_p_class)

Y_t_class = test_generator.classes
Y_t_class_categorical = keras.utils.to_categorical(Y_t_class,num_classes=noClass, dtype='float32')

print(accuracy_score(Y_t_class, Y_p_class))


# In[24]:


#Classification Report
print(classification_report(Y_t_class, Y_p_class, digits=6))


# In[25]:


cm1 = confusion_matrix(Y_t_class, Y_p_class)
print('Confusion Matrix : \n', cm1)


# In[15]:


# idx = 14
# list_of_imgs = glob(test_data_dir+'\\**\\**')

# print(len(list_of_imgs))


# In[16]:




# temp_alp =  cv2.resize(cv2.imread(list_of_imgs[idx]),(IMAGE_WIDTH, IMAGE_HEIGHT))



# plt.imshow(temp_alp)
# plt.title('Selected image')

# try:
 
#     print ('Weights are Loading............................')
#     model.load_weights(filepath)
#     print ('Weights loading Done!!!!!......................')

# except:
#     print('No pretrained weight found')
  
  
# # model.predict(temp_alp)

# temp_alp=temp_alp.astype(np.float32)
# temp_alp = temp_alp/temp_alp.max()
# img=np.expand_dims(temp_alp, axis=0)
# predict_class=model.predict(img,verbose=1)


# # print(predict_class)
# predict_class = np.argmax(predict_class, axis=1)
# # print(predict_class)


# print('Detected character is \n')
 
# if predict_class == 0:
#     print('E')

# elif predict_class ==1:
#     print('L')
  
# elif predict_class ==2:
#     print('M')
# elif predict_class ==3:
#     print('P')
  
# elif predict_class ==4:
#     print('X')    

# elif predict_class ==5:
#     print('Z')  


# In[17]:


# filepath


# In[1]:


model.load_weights(filepath)


# In[8]:


cf_matrix= [[322,,  90,  10,  33,   5 ,  0,   6,   0,  12,  18,  86,  0,   4,   6, 5,
 [  37,1969,   7,  43,   1,  11,   8,   0,  84,  27,  26,   1,   5,  33,
    68],
 [   7,  12,2267,  77,  11,  18,  22,   0,  13,  30,  61,   0,   7,  10,
    62],
 [   4,  20,  21,2687,   3,  89,  13,   0,  14,  57,  26, 689,  19,  74,
   380],
 [   1,  10 , 72,  36,1744,  41,   6,   0,   8,  19,  71,   0,   2,   6,
    68],
 [   1 ,  1,   8,  29,   4,2196,   0,   0,   9,   9,  16,   1,   0,  19,
    88],
 [   0,   4,   6,  30,   9,  31,4342,  51,  24,  43,   4,  0,74,  37,
    23],
 [   0,   0,   1,  10,   0,   4,  69, 107, 115,   5,   0,   0,   1,  91,
     8],
 [   5   36    5   54    1   11   22    0 3114  158   30    1    8   64
    61],
 [  13,  18,   9,  40,   5,   6,   8,   0, 536, 850,  15,   0,   2,  83,
    91],
 [   5,   3,  17,  29,   0,   1,  3,   0,  17,  15,1116,   0,   3,  2,
    50],
 [   0,   0,   7, 439,   2,  28,   1,   0,   7,   9,  14, 541,   0,  16,
    98],
 [   7,   3,  15,  34,  4,  16,  90,   0,  29,  23,  19,   0,2049,  32,
    33],
 [   1,   4,   2,  33,   3,  17,  56,   0,  19, 117,   4,   1,   6,4931,
    81],
 [   6,  17,  19, 110,   4,  23,  16,   0,  19,  58, 206,   0,  28,  30,
  3533]]


# In[29]:


labels = ['C','D','E','F','G','H','I','J','K','L','M','N','O',"P",'Q']
labels = np.asarray(labels).reshape(14,14)
ax = sns.heatmap(cm1, annot=labels, fmt='', cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
## Display the visualization of the Confusion Matrix.
plt.show()


# In[34]:


from sklearn.metrics import confusion_matrix

#Generate the confusion matrix
cf_matrix = confusion_matrix(Y_t_class, Y_p_class)


print(cf_matrix)


# In[35]:


import seaborn as sns
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
## Display the visualization of the Confusion Matrix.
plt.show()


# In[ ]:




