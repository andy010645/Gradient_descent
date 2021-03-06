import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
import random as random
import csv
import time
<<<<<<< HEAD
=======

>>>>>>> 7b6dd692ea5eee0914a210147cf0d428d921ae91
#create some data
x_data=[338, 333, 328, 207, 226, 25, 179, 60, 208, 606]

y_data=[640, 633, 619, 393, 428,27, 193, 66, 226, 1591]

<<<<<<< HEAD

=======
>>>>>>> 7b6dd692ea5eee0914a210147cf0d428d921ae91
x = np.arange(-200,-100,1) #bias
y = np.arange(-5,5,0.1) #weight
Z =  np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y)
for i in range(len(x)):
    for j in range(len(y)):
         b = x[i]
         w = y[j]
         Z[j][i] = 0
         for n in range(len(x_data)):
            Z[j][i] = Z[j][i] +  (y_data[n] - b - w*x_data[n])**2
            Z[j][i] = Z[j][i]/len(x_data)
<<<<<<< HEAD
            print(Z)
            #time.sleep(3)
            

#initial weight and bias
b=-120
w=-4
=======

#initial weight and bias
b=-110
w=0
>>>>>>> 7b6dd692ea5eee0914a210147cf0d428d921ae91
lr=1
iteration=100000

b_history=[b]
w_history=[w]
<<<<<<< HEAD
=======

>>>>>>> 7b6dd692ea5eee0914a210147cf0d428d921ae91
lr_b=0.0
lr_w=0.0


for i in range(iteration):
#gradient fxn
    b_grad=0.0   # 新的b點位移預測
    w_grad=0.0   # 新的w點位移預測
    
    for n in range(len(x_data)):

        # L(w,b)對b偏微分
<<<<<<< HEAD
        b_grad = b_grad -2.0*(y_data[n] - b - w*x_data[n])
=======
        b_grad = b_grad -2.0*(y_data[n] - b - w*x_data[n])*1.0
>>>>>>> 7b6dd692ea5eee0914a210147cf0d428d921ae91
        # L(w,b)對w偏微分
        w_grad = w_grad -2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
        
    lr_b = lr_b + b_grad **2
    lr_w = lr_w + w_grad **2

    b = b - lr/np.sqrt(lr_b)*b_grad # Adagrad
    w = w - lr/np.sqrt(lr_w)*w_grad
   

    b_history.append(b)
    w_history.append(w)  #put all the b,w into the array, find the minimum
<<<<<<< HEAD
=======
    
    
print(b,'\t',w,'\n',b_grad,'\t',w_grad,'\n',lr/np.sqrt(lr_b)*b_grad,'\t',lr/np.sqrt(lr_w)*w_grad,end='\n')

>>>>>>> 7b6dd692ea5eee0914a210147cf0d428d921ae91



#plot the figure

 
<<<<<<< HEAD
plt.contourf(x,y,Z, 50, alpha=0.5,cmap=plt.get_cmap('jet'))    
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
=======
plt.contourf(x,y,Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))    
#plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
>>>>>>> 7b6dd692ea5eee0914a210147cf0d428d921ae91
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()
