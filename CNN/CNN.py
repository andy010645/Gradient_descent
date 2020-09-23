import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
import sys
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data

def load_data():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    number=60000
    x_train=x_train[0:number]
    y_train=y_train[0:number]
    x_train=x_train.reshape(number,1,28,28)
    x_test=x_test.reshape(x_test.shape[0],1,28,28)
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')

    y_train=np_utils.to_categorical(y_train,10)
    y_test=np_utils.to_categorical(y_test,10)

    x_train=x_train/255
    x_test=x_test/255
    return (x_train,y_train),(x_test,y_test)

(x_train,y_train),(x_test,y_test)=load_data()

def CNN_2level():
    model = Sequential()
    model.add(Conv2D(50,3,input_shape=(1,28,28),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=2,data_format='channels_first'))
    model.add(Conv2D(100,3,activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=2,data_format='channels_first'))
    model.add(Flatten())
    print(model.summary())
    model.add(Dense(units=100,activation='relu'))
    model.add(Dense(units=10,activation='softmax'))
    # compile
    model.compile(loss=categorical_crossentropy,optimizer='Adam',metrics=['accuracy'])
    # run model
    model.fit(x_train,y_train,batch_size=100,epochs=30)
    # print result
    result=model.evaluate(x_test,y_test)
    test=model.evaluate(x_train,y_train)
    print("\nTrain Loss",test[0],"\nTrain ACC",test[1])
    print("\nLoss:",result[0],"\nTest ACC",result[1])
    
    predict_test = model.predict_classes(x_test).astype('int')
    print(predict_test,"\n")
    return predict_test

def CNN_MULIlevel():
    model = Sequential()
    model.add(Conv2D(100,3,input_shape=(1,28,28),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=2,data_format='channels_first'))
    model.add(Conv2D(200,3,activation='relu',padding='same'))
    model.add(Flatten())
    print(model.summary())
    model.add(Dense(units=100,activation='relu'))
    model.add(Dense(units=10,activation='softmax'))

    model.compile(loss=categorical_crossentropy,optimizer='Adam',metrics=['accuracy'])

    model.fit(x_train,y_train,batch_size=100,epochs=30)
    result=model.evaluate(x_test,y_test)
    test=model.evaluate(x_train,y_train)
    print("\nTrain Loss",test[0],"\nTrain ACC",test[1])
    print("\nLoss:",result[0],"\nTest ACC",result[1])


a=sys.argv[1]
print(a)
if a==1:
    print("CNN_2level")
    CNN_2level()
elif a==2:
    print("CNN_MULIlevel")
    CNN_MULIlevel()


predict_test = CNN_2level()

# PRINT ERROR
for i in range(10000):
    if  list(y_test[i]).index(1)!=predict_test[i]:
        print("TRAIN RESULT:",predict_test[i],"  TEST:",list(y_test[i]).index(1),"\n")
        image=x_test[i]
        image=np.array(image,dtype='float')
        pixels = image.reshape((28,28))
        plt.imshow(pixels, cmap='gray_r')
        plt.show()




