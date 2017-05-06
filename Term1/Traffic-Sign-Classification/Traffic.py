import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import cv2

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file= 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'


with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
import pandas as pd
df = pd.read_csv('signnames.csv')
n_classes = df.shape[0]

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

def showOneRandomImage(X, Y):
    #plot the image
    index = random.randint(0,len(X))
    image = X[index].squeeze()
    plt.figure(figsize=(2,2))
    plt.imshow(image)
    sign_index = Y[index]
    print('Sign index: {}, {}'.format(sign_index,df.iat[sign_index,1]))
    return index

def showallUniqueSigns(X_train,y_train):
    plt.figure(figsize=(10,10))
    i = 1
    labels = list(y_train)
    for label in np.arange(n_classes):
        image = X_train[labels.index(label)]
        plt.subplot(7,7,i)
        plt.axis('off')
        plt.title('Label{}'.format(label))
        i += 1
        plt.imshow(image)
    plt.show()
        
#plot hte count
def showSignHistorgraph(y):
    plt.figure()
    count = [0]*n_classes
    for i in y:
        count[i] = count[i] + 1
    y_pos = np.arange(len(count))
    plt.bar(y_pos,count,align='center',alpha = 0.5)
    plt.xlabel('Traffic Sign Index No.')
    plt.ylabel('Count')
    plt.title('Traffic Sign Count in data')
    plt.show()
    return np.array(count)
def normalize(X):
    return (X-122.5)/122.5
def RGB_GRAY(X):
    return np.expand_dims(np.average(normalize(X),axis = 3), axis=3)
def transform(image,ang, trans):
    #rotation matrix
    ar = random.uniform(-ang, ang)
    height, width, channels = image.shape
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), ar, 1)
    img = cv2.warpAffine(image, rot_mat, (width, height))
    #translation matrix
    tr_x = random.uniform(-trans,trans)
    tr_y = random.uniform(-trans,trans)
    tra_mat = np.array([[1,0,tr_x],[0,1,tr_y]])
    img2 = cv2.warpAffine(img, tra_mat, (width, height))
    return img2
def data_aug(img):
    ar = 90
    tr = 5
    image = transform(img,ar, tr)
    return image
augdata = 'aug_data.p'
with open(augdata, mode='rb') as f:
    aug = pickle.load(f)
x_aug, y_aug = aug['features'], aug['labels']
showSignHistorgraph(y_aug)
##from sklearn.utils import shuffle
##X_train, y_train = shuffle(X_train, y_train)
#show sign board meanings
#print(df.values)
#This is to show one random image from the data set
##index = random.randint(0,len(X_train))
##image = X_train[index].squeeze()
##plt.subplot(1,3,1)
##plt.imshow(image)
##plt.title('color image')
##img = np.average(X_train[index],axis = 2)
##plt.subplot(1,3,2)
##plt.imshow(img.squeeze())
##plt.title('gray_scaled')
##img2 = np.average(normalize(X_train[index]),axis = 2)
##plt.subplot(1,3,3)
##plt.imshow(img2.squeeze())
##plt.title('normalized')
##plt.show()

#this is to show all signboards from the data set
#showallUniqueSigns(X_train,y_train)
#This is to show the hostorgram distribution of the images
#label_count = showSignHistorgraph(y_train)

