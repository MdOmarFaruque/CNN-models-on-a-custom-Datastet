#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Output


# In[2]:


import cv2
import os
from glob import *


# In[3]:


list_of_dirs = glob('./data/*/**.mp4')
list_of_dirs


# In[4]:


list_of_dirs[0].split('\\')


# In[5]:


# alp_name = list_of_dirs[0].split('\\')[1]

# d_name = 'frame/'+str(alp_name)

# try:
#     os.mkdir(d_name)
    
# except:
#     pass


# ['./data', 'E', 'zoom_0.mp4']


# In[ ]:





# In[6]:


# Importing all necessary libraries
import cv2
import os

currentframe = 0


for files in list_of_dirs:
    
    print('.'*25)
    print(f'...... file name is {files} ................')
    print('.'*25)
    flg_d =1 
    
    if flg_d:
        alp_name = files.split('\\')[1]

        d_name = 'frame/'+str(alp_name)

        try:
            os.mkdir(d_name)

        except:

#             flg_d = 0
            pass




    cam = cv2.VideoCapture(files)
    
    
    
 
    

    while(True): 

        ret,frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = f'./frame/{alp_name}/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break


#         if currentframe ==5:
#             break


    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    


# In[7]:


# # Importing all necessary libraries
# import cv2
# import os

# # Read the video from specified path
# cam = cv2.VideoCapture("data/L/zoom_0.mp4")

 
# currentframe = 0

# while(True): 

#     ret,frame = cam.read()

#     if ret:
#         # if video is still left continue creating images
#         name = './data/frame/frame' + str(currentframe) + '.jpg'
#         print ('Creating...' + name)

#         # writing the extracted images
#         cv2.imwrite(name, frame)

#         # increasing counter so that it will
#         # show how many frames are created
#         currentframe += 1
#     else:
#         break
        
        
#     if currentframe ==20:
#         break
        

# # Release all space and windows once done
# cam.release()
# cv2.destroyAllWindows()


# In[8]:


currentframe #976


# In[ ]:




