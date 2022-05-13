import numpy as np
import random

class Fuzzy_Artmap():
    def __init__(self, M = 625, choice = 0.001, learn = 0.5, vig = 0.75, st_vig = 0.75, eps = 0.001):
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
        I = self.make_input(I)
           
        for index, a_i in enumerate(I):
            #just to see how fast training goes
            if index%1000==0:
                print(index)

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
                        self.W[J] = self.learn*(self.min_array(a_i,self.W[J])) + (1-self.learn)*(self.W[J])
                        break
                    else:
                        T_list[J] = 0
                        #making better vig, as we failed
                        self.vig = sum(self.min_array(a_i,self.W[J]))/self.M + self.eps
                else:
                    T_list[J] = 0
            self.vig=self.st_vig

    def predict(self, input_i):
        I =self.make_input(input_i)
        sum_reson = 0
        ans = []
        for I_i in I:
            T_list=np.array([self.choice_function(I_i, w_i) for w_i in self.W])
            while 1:
                if sum(T_list)==0:
                    ans.append(-1)
                    break;
                J = np.argmax(T_list)
                resonance = sum(self.min_array(I_i,self.W[J]))/self.M
                sum_reson += resonance
                if resonance >= self.vig:
                    ans.append(self.C[J])
                    break
                else:
                    T_list[J]=0
        return ans, sum_reson


A = Fuzzy_Artmap()
testing_data = np.array([[(i%2)/2, (i%3)/3, (i%5)/5] for i in range(10001)])
testing_answers = np.array([i % 30 for i in range(10001)])
A.train(testing_data, testing_answers)
test = np.array([random.randint(1,10000) for i in range(1000)])
test_predict = np.array([[(i%2)/2, (i%3)/3, (i%5)/5] for i in test])
res, sum_reson = A.predict(test_predict)
good, bad, undef = 0,0,0
for i in range(1000):
    if res[i] == -1:
        undef += 1
    else:
        if test[i] % 30 == res[i]:
            good += 1
        else:
            bad += 1
print("good:", good, "bad:", bad, "undef:", undef, "avg_reson:", sum_reson/1000)



B = Fuzzy_Artmap()
testing_data = np.array([[(i%2)/2, (i%3)/3, (i%5)/5, (i%7)/7, i/10001] for i in range(10001)])
testing_answers = np.array([i % 30 for i in range(10001)])
B.train(testing_data, testing_answers)
test = np.array([random.randint(1,10000) for i in range(1000)])
test_predict = np.array([[(i%2)/2, (i%3)/3, (i%5)/5, (i%7)/7, i/10001] for i in test])
res, sum_reson = B.predict(test_predict)
good, bad, undef = 0,0,0
for i in range(1000):
    if res[i] == -1:
        undef += 1
    else:
        if test[i] % 30 == res[i]:
            good += 1
        else:
            bad += 1
print("good:", good, "bad:", bad, "undef:", undef, "avg_reson:", sum_reson/1000)



C = Fuzzy_Artmap()
testing_data = np.array([[random.randint(1, 10)/10 for i in range(5)] for i in range(5001)])
testing_answers = np.array([np.argmax(testing_data[i]) for i in range(5001  )])
C.train(testing_data, testing_answers)
test_predict = np.array([[random.randint(1, 10)/10 for i in range(5)] for i in range(1000)])
res, sum_reson = C.predict(test_predict)
good, bad, undef = 0,0,0
for i in range(1000):
    if res[i] == -1:
        undef += 1
    else:
        for j in range(5):
            if test_predict[i][j] > test_predict[i][res[i]]:
                bad += 1
                break
        else:
            good += 1
print("good:", good, "bad:", bad, "undef:", undef)



import matplotlib.pyplot as plt
import os
from matplotlib.image import imread
%matplotlib inline
import os
A = imread('first/l1_n1_x2987_y3748.tif')
print(A.shape)
_ = plt.imshow(A, cmap='gray')



data = [[] for i in range(8)]
for root, dirs, files in os.walk("./first"):
    for filename in files:
        cl = int(str(filename)[1]) - 1
        A = imread("first/" + str(filename))
        for i in range(6):
            for j in range(6):
                curA = A[25*i:25*(i+1), 25*j:25*(j+1)][:,:,:-1]
                resA = np.zeros((5,5,3))
                for x in range(5):
                    for y in range(5):
                        for ix in range(5*x, 5*(x+1)):
                            for iy in range(5*y, 5*(y+1)):
                                resA[x][y] +=curA[ix][iy]
                resA /= (25*256)
                data[cl].append(resA)
for root, dirs, files in os.walk("./second"):
    for filename in files:
        cl = int(str(filename)[1]) - 1
        A = imread("second/" + str(filename))
        for i in range(6):
            for j in range(6):
                curA = A[25*i:25*(i+1), 25*j:25*(j+1)][:,:,:-1]
                resA = np.zeros((5,5,3))
                for x in range(5):
                    for y in range(5):
                        for ix in range(5*x, 5*(x+1)):
                            for iy in range(5*y, 5*(y+1)):
                                resA[x][y] +=curA[ix][iy]
                resA /= (25 * 256)
                data[cl].append(resA)



print([len(data[i]) for i in range(8)])
print(sum([len(data[i]) for i in range(8)]))



resulted_data = np.zeros((13680, 76))
s=0
for cl in range(8):
    for j in data[cl]:
        resulted_data[s][:-1] = np.reshape(j, (75, ))
        resulted_data[s][-1] = cl
        s+=1
print(resulted_data)



np.random.shuffle(resulted_data)
final_data = resulted_data[:,:-1]
final_data_answers = resulted_data[:, -1]
print(final_data)
print(final_data_answers)
with open('test_data.npy', 'wb') as f:
    np.save(f, final_data)
with open('test_data_answers.npy', 'wb') as f:
    np.save(f, final_data_answers)



A = Fuzzy_Artmap()
testing_data = np.load("./test_data.npy")
testing_answers = np.load("./test_data_answers.npy")
print(testing_data.shape, testing_answers.shape)
print(testing_data.shape)
A.train(testing_data[:10000], testing_answers[:10000])



test_predict = testing_data[10000:11000]
res, sum_reson = A.predict(test_predict)
good, bad, undef = 0,0,0
match_array = np.zeros((8,8))
for i in range(1000):
    if res[i] == -1:
        undef += 1
    else:
        match_array[int(testing_answers[10000+i])][int(res[i])]+=1
        if testing_answers[10000+i] == res[i]:
            good += 1
        else:
            bad += 1
print("good:", good, "bad:", bad, "undef:", undef)
print(match_array)