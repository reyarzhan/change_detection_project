import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

class Fuzzy_Artmap():
    def __init__(self, M = 625, choice = 0.001,
                 learn = 0.5, vig = 0.75, st_vig = 0.75,eps = 0.001):
        #size of F-level
        self.M = M 
        #choice parameter
        self.choice=choice
        #learning parameter 
        self.learn=learn
        #changable vigilance parameter
        self.vig=vig
        #stable vigilance parameter
        self.st_vig=st_vig
        #matching parameter
        self.eps=eps
        #weights from F to C
        self.C=np.array([])
        #weights from I to F
        self.W=np.array([])
        #original data answers
        self.orig_result =np.array([])
        #size of W
        self.sz_W = np.array([])

    #complete our input
    def make_input(self, I):
        return np.concatenate((I,np.ones(I.shape) - I),axis=1)
    
    #func to calculate a^b
    def min_array(self,a,b):
        return np.array([min(a[i],b[i]) for i in range(len(a))])
    
    #func with which we select the optimal F-level vertex for our input
    def choice_function(self,x,w):
        T = sum(self.min_array(x,w))/(self.choice + sum(w))
        return T
    

    def train(self,I, I_res):
        self.M = len(I[0])
        self.orig_result=I_res
        self.C = [0]
        self.W = np.ones((1,self.M*2))
        self.sz_W = np.array([self.W.shape[0]])
        I = self.make_input(I)
        for index, a_i in enumerate(I):
            if index==0:
                self.W[0]=a_i
                self.C[0]=self.orig_result[0];
                continue
            T_list=np.array([self.choice_function(a_i, w_i) for w_i in self.W])
            T_max = np.argmax(T_list)
            while 1:
                #if no good F-vertex found, build a new one
                if sum(T_list)==0:
                    #if self.W.shape[0]<400:
                    self.C = np.concatenate((self.C,[self.orig_result[index]]))
                    self.W = np.concatenate((self.W,[a_i]),axis=0)
                    #else:
                    #    self.C[T_max] = self.orig_result[index]
                    break

                #best matching F-vertex
                J = np.argmax(T_list)
                res = sum(self.min_array(a_i,self.W[J]))/self.M 
                if res >= self.vig:
                    if self.orig_result[index]==self.C[J]:
                        #making new weights
                        self.W[J] = self.learn*(self.min_array(a_i,self.W[J])) 
                        + (1-self.learn)*(self.W[J])
                        break
                    else:
                        T_list[J] = 0
                        #making better vig, as we failed
                        self.vig = sum(self.min_array(a_i,self.W[J]))/self.M + self.eps
                else:
                    T_list[J] = 0
            self.vig = self.st_vig
            self.sz_W = np.concatenate((self.sz_W,[self.W.shape[0]]),axis=0)

    def predict(self, input_i):
        I =self.make_input(input_i)
        ans = []
        for I_i in I:
            T_list=np.array([self.choice_function(I_i, w_i)
                             for w_i in self.W])
            while 1:
                if sum(T_list)==0:
                    ans.append(-1)
                    break;
                J = np.argmax(T_list)
                resonance = sum(self.min_array(I_i,self.W[J]))/self.M
                if resonance >= self.vig:
                    ans.append(self.C[J])
                    break
                else:
                    T_list[J]=0
        return ans

def classification(file1):
    A = imread(file1)
    A_data = []
    for i in range(6):
        for j in range(6):
            curA = A[25*i:25*(i+1), 25*j:25*(j+1)]
            curA = curA[:, : , :-1]
            resA = np.zeros((5,5,3))
            for x in range(5):
                for y in range(5):
                    for ix in range(5*x, 5*(x+1)):
                        for iy in range(5*y, 5*(y+1)):
                            resA[x][y] +=curA[ix][iy]
            resA /= (25 * 256)
            A_data.append(resA)
    resulted_data_A = np.zeros((36, 75))
    s=0
    for j in A_data:
        resulted_data_A[s] = np.reshape(j, (75, ))
        s+=1

    resA = F.predict(resulted_data_A)
    return np.array(resA) / 8


def change_detection(file1, file2): 
    resA = classification(file1)
    resB = classification(file2)
    ansA = np.zeros((6,6,3))
    ansB = np.zeros((6,6,3))
    change = np.zeros((6,6,3))
    for i in range(6):
        for j in range(6):
            ansA[i,j]=np.array([resA[6*i+j],resA[6*i+j],resA[6*i+j] ])
            ansB[i,j]=np.array([resB[6*i+j],resB[6*i+j],resB[6*i+j] ])
            if ansA[i,j][0] == ansB[i][j][0]:
                change[i,j] = np.array([1.,1.,1.])
            else:
                change[i,j] = np.array([0.,0.,0.])

    return change



F = Fuzzy_Artmap()
testing_data = np.load("./test_data.npy")
testing_answers = np.load("./test_data_answers.npy")
F.train(testing_data[:500], testing_answers[:500])

t1 = change_detection("l8_n165_x3404_y3299.tif", "l8_n173_x3404_y3292.tif")
print(t1[:,:,0])