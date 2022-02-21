import matplotlib.pyplot as plt
import cv2
from PIL import Image


plt.figure(figsize=(4, 4), num = 'conv1')
for i in range(16):
    path = 'conv1/'+ str(i) + '_256.png'
    img = Image.open(path)
    plt.subplot(4, 4, i+1)
    plt.imshow(img)
    plt.axis('off')
plt.savefig('conv1.png')



plt.figure(figsize=(6,6), num = 'conv2')
for i in range(32):
    path = 'conv2/'+ str(i) + '_256.png'
    img = Image.open(path)
    plt.subplot(6, 6, i+1)
    plt.imshow(img)
    plt.axis('off')
plt.savefig('conv2.png')


plt.figure(figsize=(8,8), num = 'conv3')
for i in range(64):
    path = 'conv3/'+ str(i) + '_256.png'
    img = Image.open(path)
    plt.subplot(8, 8, i+1)
    plt.imshow(img)
    plt.axis('off')
plt.savefig('conv3.png')


plt.figure(figsize=(8,8), num = 'conv4')
for i in range(64):
    path = 'conv4/'+ str(i) + '_256.png'
    img = Image.open(path)
    plt.subplot(8, 8, i+1)
    plt.imshow(img)
    plt.axis('off')
plt.savefig('conv4.png')


# plt.show()