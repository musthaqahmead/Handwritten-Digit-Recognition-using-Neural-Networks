#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2


# In[8]:


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train.shape


# In[9]:


plt.imshow(x_train[0])
plt.show()
plt.imshow(x_train[0], cmap=plt.cm.binary)


# In[10]:


print (x_train[0])


# In[11]:


x_train=tf.keras.utils.normalize(x_train, axis=1)
x_test=tf.keras.utils.normalize(x_test, axis=1)
plt.imshow(x_train[0], cmap=plt.cm.binary)


# In[12]:


print(x_train[0])


# In[13]:


print(y_train[0])


# In[14]:


IMG_SIZE=28
x_trainr=np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE,1)
x_testr=np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE,1)
print("Training Samples dimension", x_trainr.shape)
print("Testing Samples dimension",x_testr.shape)


# In[15]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


# In[16]:


model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add (Flatten())
model.add (Dense(64))
model.add(Activation("relu"))

model.add (Dense(32))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation('softmax'))


# In[17]:


model.summary()


# In[18]:


print("Total Training Samples = ",len(x_trainr))


# In[19]:


model.compile(loss ="sparse_categorical_crossentropy", optimizer ="adam", metrics=['accuracy'])


# In[20]:


model.fit(x_trainr,y_train,epochs=5, validation_split=0.3)


# In[21]:


test_loss, test_acc = model.evaluate(x_testr, y_test)
print("test Loss on 10,000 test samples", test_loss)
print("Validation Accuracy on 10,000 test samples", test_acc)


# In[22]:


predictions=model.predict([x_testr])


# In[23]:


print(predictions)


# In[24]:


print(np.argmax(predictions[0]))


# In[25]:


plt.imshow(x_test[0])


# In[26]:


print(np.argmax(predictions[128]))


# In[27]:


plt.imshow(x_testr[128])


# In[71]:


img = cv2.imread(r'C:\Users\user\Desktop\six.png')

if img is None:
    print("Error: Image not found or unable to load.")
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.axis('on')
    plt.show()


# In[72]:


img.shape


# In[73]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[74]:


gray.shape


# In[75]:


resized = cv2.resize(gray, (28,28),interpolation = cv2.INTER_AREA)


# In[76]:


resized.shape


# In[77]:


newimg = tf.keras.utils.normalize (resized, axis=1)


# In[78]:


newimg.shape


# In[79]:


newimg = newimg.reshape(1, 28, 28, 1)
predictions = model.predict(newimg)


# In[80]:


print(np.argmax(predictions))


# In[ ]:




