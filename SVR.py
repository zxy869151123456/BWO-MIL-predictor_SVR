import numpy as np
from Normalize import FeatureScaler
# from Normalize import FeatureScaler

# debug import
import pandas as pd

class CSet():
    def __init__(self,size):
        self.m_indicators = [False] * size
        self.m_next = np.zeros((size),dtype='int')
        self.m_previous = np.zeros((size),dtype='int')
        self.m_number = 0
        self.m_first = -1

    def contains(self,index):
        return self.m_indicators[index]

    def Delete(self,index):
        if self.m_indicators[index]:
            if self.m_first == index:
                self.m_first = self.m_next[index]
            else:
                self.m_next[self.m_previous[index]] = self.m_next[index]

            if self.m_next[index] != -1:
                self.m_previous[self.m_next[index]] = self.m_previous[index]
            self.m_indicators[index] = False
            self.m_number -= 1

    def insert(self,index):
        if not self.m_indicators[index]:
            if self.m_number == 0:
                self.m_first = index
                self.m_next[index] = -1
                self.m_previous[index] = -1
            else:
                self.m_previous[self.m_first] =index
                self.m_next[index] = self.m_first
                self.m_previous[index] = -1
                self.m_first = index
            self.m_indicators[index] = True
            self.m_number += 1
    def getNext(self,index):
        if index == -1:
            return self.m_first
        else:
            return self.m_next[index]


class SVR():
    def __init__(self,task_info):
        self.pr_kernal_type = task_info['kernel_type']
        self.m_epsilon = np.float32(task_info['epsilon'])
        self.m_c = np.float32(task_info['cost'])
        self.pr_rbf_gamma = np.float32(task_info['rbf_gamma'])
        self.pr_coef_lin = np.float32(task_info['coef_lin'])
        self.pr_coef_const = np.float32(task_info['coef_const'])
        self.pr_poly_degree = task_info['poly_degree']
        self.max_opt_times = int(task_info['max_opt_times'])

        self.pr_sample_num = None
        self.m_feature_num = None
        self.m_test = None

        self.pr_tol = np.float32(1e-4)
        self.pr_num_of_sub_QP = None
        self.pr_times_of_update = None
        self.pr_optimization_times = None

        # model part
        self.m_alph = None
        self.pr_e = None
        self.pr_c = None
        self.pr_y = None
        self.pr_xx = None
        self.m_w = None
        self.m_f = None
        self.m_b = None
        self.NumSVM = None

        self.pr_vector_a = None
        self.pr_vector_b = None
        self.pr_xxOn = True

        self.normalizer = None

    def __scaling_Y(self,Y):
        # scale the Y
        # use min max and let the Y in [0.0,0.1]
        y_min = np.min(Y, axis=0)
        y_max = np.max(Y, axis=0)

        Y = ((Y - y_min) / (y_max - y_min))
        self.y_min = y_min
        self.y_max = y_max

        return Y

    def __recovery_Y(self,Y):
        return (Y) * (self.y_max-self.y_min) + self.y_min

    def __prepare_c_and_y(self,train_data):
        X, Y = train_data

        for row in range(self.pr_sample_num):
            self.pr_c[row] = -(self.m_epsilon + Y[row])
            self.pr_c[row + self.pr_sample_num] = -(self.m_epsilon - Y[row])
            self.pr_y[row] = 1
            self.pr_y[row + self.pr_sample_num] = -1

    def __prepare_xx(self,train_data):
        X,Y = train_data

        m = self.m_feature_num

        for row in range(self.pr_sample_num):
            k1 = row * m
            for col in range(row + 1):
                k2 = col * m
                for i in range(m):
                    self.pr_vector_a[i] = X[k1 + i]
                    self.pr_vector_b[i] = X[k2 + i]
                self.pr_xx[row * self.pr_sample_num + col] = self.__kernel(self.pr_vector_a,self.pr_vector_b)
                self.pr_xx[col * self.pr_sample_num + row] = self.pr_xx[row * self.pr_sample_num + col]

    def __kernel(self,a,b):
        if self.pr_kernal_type == 'linear':
            return self.__sprod_ss(a,b)
        elif self.pr_kernal_type == 'polynomial':
            return np.power(self.pr_coef_lin * self.__sprod_ss(a,b) + self.pr_coef_const,self.pr_poly_degree)
        elif self.pr_kernal_type == 'rbf':
            return np.exp(-self.pr_rbf_gamma * (self.__sprod_ss(a,a) - 2 * self.__sprod_ss(a,b) + self.__sprod_ss(b,b)))
        elif self.pr_kernal_type == 'sigmoid':
            return np.tanh(self.pr_coef_lin * self.__sprod_ss(a,b) + self.pr_coef_const)

    def __sprod_ss(self,a,b):
        sum = 0.0
        for i in range(self.m_feature_num):
            sum += a[i] * b[i]
        return sum

    def __examineExample(self,i2):
        alpha2 = self.m_alph[i2]
        alpha2_ = self.m_alph[self.pr_sample_num + i2]

        if i2 < 0:
            return  0
        F2 = np.float32(0.0)
        if self.m_I0.contains(i2):
            F2 = self.m_f[i2]
        else:
            F2 = self.pr_training_set[1][i2]
            for j in range(self.pr_sample_num):
                F2 -= (self.m_alph[j] - self.m_alph[self.pr_sample_num + j]) * self.__xx(i2,j)
            self.m_f[i2] = F2

            if self.m_I1.contains(i2):
                if F2 + self.m_epsilon < self.m_bUp:
                    self.m_bUp = F2 + self.m_epsilon
                    self.m_iUp = i2
                elif F2 - self.m_epsilon > self.m_bLow:
                    self.m_bLow = F2 - self.m_epsilon
                    self.m_iLow = i2
            elif self.m_I2.contains(i2) and (F2 + self.m_epsilon > self.m_bLow):
                self.m_bLow = F2 + self.m_epsilon
                self.m_iLow = i2
            elif self.m_I3.contains(i2) and (F2 - self.m_epsilon < self.m_bUp):
                self.m_bUp = F2 - self.m_epsilon
                self.m_iUp = i2
        if self.m_iUp < 0:
            return 0

        optimality = True
        i1 = -1

        if (self.m_I0.contains(i2) and 0 < alpha2 and alpha2 < self.m_c):
            if (self.m_bLow - (F2 - self.m_epsilon) > 2 * self.pr_tol):
                optimality = False
                i1 = self.m_iLow
                if ((F2 - self.m_epsilon) - self.m_bUp > self.m_bLow - (F2 - self.m_epsilon)):
                    i1 = self.m_iUp
            elif ((F2 - self.m_epsilon) - self.m_bUp > 2 * self.pr_tol):
                optimality = False
                i1 = self.m_iUp
                if (self.m_bLow - (F2 - self.m_epsilon) > (F2 - self.m_epsilon) - self.m_bUp):
                    i1 = self.m_iLow
        elif (self.m_I0.contains(i2) and 0 < alpha2_ and alpha2_ < self.m_c):
            if (self.m_bLow - (F2 + self.m_epsilon) > 2 * self.pr_tol):
                optimality = False
                i1 = self.m_iLow
                if ((F2 + self.m_epsilon) - self.m_bUp > self.m_bLow - (F2 + self.m_epsilon)):
                    i1 = self.m_iUp
            elif((F2 + self.m_epsilon) - self.m_bUp > 2 * self.pr_tol):
                optimality = False
                i1 = self.m_iUp
                if(self.m_bLow - (F2 + self.m_epsilon) > (F2 + self.m_epsilon) - self.m_bUp):
                    i1 = self.m_iLow

        elif(self.m_I1.contains(i2)):
            if (self.m_bLow - (F2 + self.m_epsilon) >= 2 * self.pr_tol):
                optimality = False
                i1 = self.m_iLow
                if((F2 + self.m_epsilon) - self.m_bUp >= self.m_bLow - (F2 + self.m_epsilon)):
                    i1 = self.m_iUp
            elif ((F2 - self.m_epsilon) - self.m_bUp >= 2 * self.pr_tol):
                optimality = False
                i1 = self.m_iUp
                if (self.m_bLow - (F2 - self.m_epsilon) >= (F2 - self.m_epsilon) - self.m_bUp):
                    i1 = self.m_iLow

        elif(self.m_I2.contains(i2)):
            if ((F2 + self.m_epsilon) - self.m_bUp >= 2 * self.pr_tol):
                optimality = False
                i1 = self.m_iUp
        elif(self.m_I3.contains(i2)):
            if(self.m_bLow - (F2 - self.m_epsilon) >= 2 * self.pr_tol):
                optimality = False
                i1 = self.m_iLow
        else:
            return 0

        if optimality:
            return 0

        if (self.__takeStep(i1,i2)):
            return 1
        else:
            return 0

    def __takeStep(self,i1,i2):
        if (i1 == i2 or np.abs(i1 - i2) == self.pr_sample_num):
            return 0

        alpha1 = self.m_alph[i1]
        alpha1_ = self.m_alph[self.pr_sample_num + i1]
        alpha2 = self.m_alph[i2]
        alpha2_ = self.m_alph[self.pr_sample_num + i2]
        C1 = self.m_c
        C2 = self.m_c

        F1 = self.m_f[i1]
        F2 = self.m_f[i2]

        k11 = self.__xx(i1,i1)
        k12 = self.__xx(i1,i2)
        k22 = self.__xx(i2,i2)

        eta = np.float32(-2.0) * k12 + k11 + k22

        gamma = alpha1 - alpha1_ + alpha2 - alpha2_

        self.pr_num_of_sub_QP += 1

        if (eta < 0):
            eta=0

        case1 = False
        case2 = False
        case3 = False
        case4 = False
        finished = False

        deltaphi = F1 -F2
        changed = False

        while(not finished):
            if(not case1 and (alpha1 > 0 or (alpha1_ == 0 and deltaphi > 0)) and (alpha2 > 0 or (alpha2_ ==0 and deltaphi<0))):
                L = np.maximum(0,gamma-C1)
                H = np.minimum(C2,gamma)

                if (L < H):
                    if(eta > 0):
                        a2 = alpha2 - (deltaphi / eta)
                        if (a2 > H):
                            a2 = H
                        elif (a2 < L):
                            a2 = L
                    else:

                        Lobj = -L * deltaphi
                        Hobj = -H * deltaphi
                        if(Lobj > Hobj):
                            a2 = L
                        else:
                            a2 = H
                    a1 = alpha1 - (a2 - alpha2)

                    if np.abs(a1 - alpha1) >= self.m_eps or np.abs(a2 - alpha2) >= self.m_eps:
                        alpha1 = a1
                        alpha2 = a2
                        changed = True
                else:
                    finished = True
                case1 = True
            elif (not case2 and (alpha1 > 0 or (alpha1_== 0 and deltaphi >= 2 * self.m_epsilon)) and (alpha2_ > 0 or (alpha2 == 0 and deltaphi >= 2 * self.m_epsilon))):
                L = np.maximum(0,-gamma)
                H = np.minimum(C2,-gamma + C1)

                if (L < H):
                    if(eta > 0):
                        a2 = alpha2_ + ((deltaphi - 2 * self.m_epsilon) / eta)
                        if (a2 > H):
                            a2 = H
                        elif(a2 < L):
                            a2 = L
                    else:
                        Lobj = L * (-2 * self.m_epsilon + deltaphi)
                        Hobj = H * (-2 * self.m_epsilon + deltaphi)
                        if (Lobj > Hobj):
                            a2 = L
                        else:
                            a2 = H
                    a1 = alpha1 + (a2 -alpha2_)
                    if(np.abs(a1 - alpha1) > self.m_eps or np.abs(a2 - alpha2_) > self.m_eps):
                        alpha1 = a1
                        alpha2_ = a2
                        changed = True
                    else:
                        finished = True
                    case2 = True
            elif (not case3 and (alpha1_ > 0 or (alpha1 == 0 and deltaphi < -2 * self.m_epsilon)) and (alpha2 > 0 or (alpha2_ == 0 and deltaphi < -2 * self.m_epsilon))):
                L = np.maximum(0,gamma)
                H = np.minimum(C2,C1 + gamma)

                if(L <H):
                    if(eta > 0):
                        a2 = alpha2 - ((deltaphi + 2 * self.m_epsilon) / eta)
                        if (a2 > H):
                            a2 = H
                        elif (a2 < L):
                            a2 = L
                    else:
                        Lobj = -L * (2 * self.m_epsilon + deltaphi)
                        Hobj = -H * (2 * self.m_epsilon + deltaphi)
                        if (Lobj > Hobj):
                            a2 = L
                        else:
                            a2 = H
                    a1 = alpha1_ + (a2 - alpha2)
                    if (np.abs(a1 - alpha1_) > self.m_eps or np.abs(a2 - alpha2) > self.m_eps):
                        alpha1_ = a1
                        alpha2 = a2
                        changed = True
                else:
                    finished = True
                case3 = True
            elif (not case4 and (alpha1_ > 0 or (alpha1 == 0 and deltaphi < 0)) and (alpha2_ > 0 or (alpha2 == 0 and deltaphi>0))):
                L = np.maximum(0,-gamma-C1)
                H = np.minimum(C2, -gamma)

                if(L < H):
                    if (eta > 0):
                        a2 = alpha2_ + deltaphi / eta
                        if(a2 > H):
                            a2 = H
                        elif(a2 <L):
                            a2 = L
                    else:
                        Lobj = L * deltaphi
                        Hobj = H * deltaphi
                        if (Lobj > Hobj):
                            a2 = L
                        else:
                            a2 = H
                    a1 = alpha1_ - (a2 -alpha2_)
                    if (np.abs(a1 -alpha1_) > self.m_eps or np.abs(a2 - alpha2_) > self.m_eps):
                        alpha1_ = a1
                        alpha2_ = a2
                        changed = True
                else:
                    finished = True
                case4 = True
            else:
                finished = True

            deltaphi += eta * ((alpha2 - alpha2_) - (self.m_alph[i2] - self.m_alph[self.pr_sample_num + i2]))

        if changed:
            i = self.m_I0.getNext(-1)
            while True:
                if(i != i1 and i != i2):
                    self.m_f[i] += ((self.m_alph[i1] - self.m_alph[self.pr_sample_num + i1]) - (alpha1 - alpha1_)) * self.__xx(i1,i) + ((self.m_alph[i2] - self.m_alph[self.pr_sample_num + i2]) - (alpha2 - alpha2_)) * self.__xx(i2,i)
                i = self.m_I0.getNext(i)
                if i == -1:
                    break

            self.m_f[i1] += ((self.m_alph[i1] - self.m_alph[self.pr_sample_num + i1]) - (alpha1 - alpha1_)) * k11 + ((self.m_alph[i2] - self.m_alph[self.pr_sample_num + i2]) - (alpha2 - alpha2_)) * k12
            self.m_f[i2] += ((self.m_alph[i1] - self.m_alph[self.pr_sample_num + i1]) - (alpha1 - alpha1_)) * k12 + ((self.m_alph[i2] - self.m_alph[self.pr_sample_num + i2]) - (alpha2 - alpha2_)) * k22

            m_Del = 1e-7

            if (alpha1 > (1-m_Del) * C1):
                alpha1 = self.m_c
            elif (alpha1 <= m_Del * C1):
                alpha1 = np.float32(0)

            if(alpha1_ > self.m_c - m_Del * self.m_c):
                alpha1_ = self.m_c
            elif (alpha1_ <= m_Del * self.m_c):
                alpha1_ = np.float32(0)

            if (alpha2 > ( 1- m_Del) * C2):
                alpha2 = self.m_c
            elif(alpha2 <= m_Del * C2):
                alpha2 = np.float32(0)

            if (alpha2_ > self.m_c - m_Del * self.m_c):
                alpha2_ = self.m_c
            elif(alpha2_ <= m_Del * self.m_c):
                alpha2_ = np.float32(0)

            # store the changes in alpha
            self.m_alph[i1] = alpha1
            self.m_alph[self.pr_sample_num + i1] = alpha1_
            self.m_alph[i2] = alpha2
            self.m_alph[self.pr_sample_num + i2] = alpha2_

            # update I_0 I_1 I_2 I_3
            if((0 < alpha1 and alpha1 < C1) or ( 0 < alpha1_ and alpha1_ < C1)):
                self.m_I0.insert(i1)
            else:
                self.m_I0.Delete(i1)

            if (alpha1 == 0 and alpha1_ == 0):
                self.m_I1.insert(i1)
            else:
                self.m_I1.Delete(i1)

            if (alpha1 == 0 and alpha1_ == C1):
                self.m_I2.insert(i1)
            else:
                self.m_I2.Delete(i1)

            if (alpha1 == C1 and alpha1_ == 0):
                self.m_I3.insert(i1)
            else:
                self.m_I3.Delete(i1)

            if((0 < alpha2 and alpha2 < C2 ) or (0 < alpha2_ and alpha2_ < C2)):
                self.m_I0.insert(i2)
            else:
                self.m_I0.Delete(i2)

            if alpha2 == 0 and alpha2_ == 0:
                self.m_I1.insert(i2)
            else:
                self.m_I1.Delete(i2)

            if alpha2 == 0 and alpha2_ == C2:
                self.m_I2.insert(i2)
            else:
                self.m_I2.Delete(i2)

            if alpha2 == C2 and alpha2_ == 0:
                self.m_I3.insert(i2)
            else:
                self.m_I3.Delete(i2)


            # compute (i_low, b_low) and (i_up, b_up) by applying the conditions
            # mentionned above, using only i1, i2 and indices in I_0
            self.m_bLow = -32767 * 32767
            self.m_bUp = 32767*32767
            self.m_iLow = -1
            self.m_iUp = -1

            i = self.m_I0.getNext(-1)
            while True:
                if ( 0 < self.m_alph[self.pr_sample_num + i] and self.m_alph[self.pr_sample_num +i] < self.m_c and (self.m_f[i] + self.m_epsilon) > self.m_bLow):
                    self.m_bLow = self.m_f[i] + self.m_epsilon
                    self.m_iLow = i
                elif ( 0< self.m_alph[i]) and (self.m_alph[i] < self.m_c) and (self.m_f[i] - self.m_epsilon > self.m_bLow):
                    self.m_bLow = self.m_f[i] - self.m_epsilon
                    self.m_iLow = i

                if (0 < self.m_alph[i] and self.m_alph[i] < self.m_c and (self.m_f[i] - self.m_epsilon < self.m_bUp)):
                    self.m_bUp = self.m_f[i] - self.m_epsilon
                    self.m_iUp = i
                elif ( 0 < self.m_alph[self.pr_sample_num + i] and self.m_alph[self.pr_sample_num + i] < self.m_c and (self.m_f[i] + self.m_epsilon < self.m_bUp)):
                    self.m_bUp = self.m_f[i] + self.m_epsilon
                    self.m_iUp = i

                i = self.m_I0.getNext(i)
                if (i == -1):
                    break
            if (not self.m_I0.contains(i1)):
                if(self.m_I2.contains(i1) and (self.m_f[i1] + self.m_epsilon > self.m_bLow)):
                    self.m_bLow = self.m_f[i1] + self.m_epsilon
                    self.m_iLow = i1
                elif (self.m_I1.contains(i1) and self.m_f[i1] - self.m_epsilon > self.m_bLow):
                    self.m_bLow = self.m_f[i1] - self.m_epsilon
                    self.m_iLow = i1

                if(self.m_I3.contains(i1) and (self.m_f[i1] + self.m_epsilon < self.m_bUp)):
                    self.m_bUp = self.m_f[i1] - self.m_epsilon
                    self.m_iUp = i1
                elif (self.m_I1.contains(i1) and self.m_f[i1] + self.m_epsilon < self.m_bUp):
                    self.m_bUp = self.m_f[i1] + self.m_epsilon
                    self.m_iUp = i1

            if (not self.m_I0.contains(i2)):
                if (self.m_I2.contains(i2) and self.m_f[i2] + self.m_epsilon > self.m_bLow):
                    self.m_bLow=self.m_f[i2] + self.m_epsilon
                    self.m_iLow = i2
                elif (self.m_I1.contains(i2) and self.m_f[i2] - self.m_epsilon > self.m_bLow):
                    self.m_bLow = self.m_f[i2] - self.m_epsilon
                    self.m_iLow = i2

                if(self.m_I3.contains([i2] and self.m_f[i2] - self.m_epsilon < self.m_bUp)):
                    self.m_bUp = self.m_f[i2] - self.m_epsilon
                    self.m_iUp = i2
                elif (self.m_I1.contains(i2) and (self.m_f[i2] + self.m_epsilon < self.m_bUp)):
                    self.m_bUp = self.m_f[i2] + self.m_epsilon
                    self.m_iUp = i2
            if (self.m_iLow == -1 or self.m_iUp == -1):
                return False
            self.pr_times_of_update += 1
            return True
        else:
            return False

    def __prepare_m_f(self):
        for row in range(self.pr_sample_num):
            self.m_f[row] = self.__f_internal(row)

    def __get_w_and_b(self,train_data):
        X,Y = train_data

        for j in range(self.m_feature_num):
            self.m_w[j] = 0

        for i in range(self.pr_sample_num):
            if ((self.m_alph[i + self.pr_sample_num] - self.m_alph[i]) != 0):
                k1 = i * self.m_feature_num
                for j in range(self.m_feature_num):
                    self.m_w[j] += (self.m_alph[i] - self.m_alph[i + self.pr_sample_num]) * X[k1 + j]

    def __f_internal(self,sample_no):
        wx = 0.0
        for i in range(self.pr_sample_num * 2):
            if self.m_alph[i] > 0:
                wx -= self.m_alph[i] * self.pr_y[i] * self.__xx(i,sample_no)
        return wx + self.m_b

    def __xx(self,x,y):
        if (x < 0 or y < 0 or x >= self.pr_sample_num * 2 or y >= self.pr_sample_num * 2):
            return 0
        if (y >= self.pr_sample_num):
            y = y -self.pr_sample_num
        if(x >= self.pr_sample_num):
            x = x - self.pr_sample_num

        if self.pr_xxOn:
            return self.pr_xx[y * self.pr_sample_num + x]
        else:
            k1 = x * self.m_feature_num
            k2 = y * self.m_feature_num
            for i in range(self.m_feature_num):
                self.pr_vector_a[i] = self.pr_training_set[0][k1 + i]
                self.pr_vector_b[i] = self.pr_training_set[0][k2 + i]
            return self.__kernel(self.pr_vector_a,self.pr_vector_b)

    def __f_external(self,x):
        wx = np.float32(0)

        for i in range(self.pr_sample_num):
            if((self.m_alph[i + self.pr_sample_num] - self.m_alph[i]) != 0):
                k1 = self.m_feature_num * i
                for j in range(0,self.m_feature_num):
                    self.pr_vector_a[j] = self.pr_training_set[0][k1 + j]
                wx += (self.m_alph[i] - self.m_alph[i + self.pr_sample_num]) * self.__kernel(self.pr_vector_a,x)
        return wx + self.m_b

    def fit(self,X,Y):

        self.normalizer = FeatureScaler(method='minmax')
        X = self.normalizer(X)

        # scale_Y
        Y = self.__scaling_Y(Y)

        self.pr_sample_num = X.shape[0]
        self.m_feature_num = X.shape[1]
        self.m_test = X.shape[1]

        self.pr_e = np.empty((self.pr_sample_num * 2),dtype='float32')
        self.pr_c = np.empty((self.pr_sample_num * 2),dtype='float32')
        self.pr_y = np.empty((self.pr_sample_num * 2),dtype='float32')

        if self.pr_sample_num > 100 :
            self.pr_xxOn = False
        else:
            self.pr_xx = np.empty((self.pr_sample_num * self.pr_sample_num),dtype='float32')

        self.m_eps = np.float32(1e-12)
        self.pr_num_of_sub_QP = 0
        self.pr_times_of_update = 0
        self.pr_optimization_times = 0

        self.m_alph = np.zeros((self.pr_sample_num * 2),dtype='float32')
        self.m_b = 0

        self.m_w = np.zeros((self.m_feature_num),dtype='float32')
        self.m_f = np.zeros((self.pr_sample_num),dtype='float32')

        self.m_I0 = CSet(self.pr_sample_num)
        self.m_I1 = CSet(self.pr_sample_num)
        self.m_I2 = CSet(self.pr_sample_num)
        self.m_I3 = CSet(self.pr_sample_num)

        for i in range(self.pr_sample_num):
            self.m_I1.insert(i)

        self.pr_vector_a = np.empty((self.m_feature_num),dtype='float32')
        self.pr_vector_b = np.empty((self.m_feature_num),dtype='float32')

        X = X.reshape((X.shape[0] * X.shape[1]))

        self.pr_training_set = (X,Y)

        self.__prepare_c_and_y((X,Y))

        if self.pr_xxOn :
            self.__prepare_xx((X,Y))

        self.m_bUp = Y[0] + self.m_epsilon + self.m_eps
        self.m_bLow = Y[0] - self.m_epsilon - self.m_eps
        self.m_iUp = 0
        self.m_iLow = 0

        numChanged = 0
        examineAll = True
        iteration = 0

        while (numChanged > 0 or examineAll) and (self.pr_optimization_times < self.max_opt_times):

            numChanged = 0

            if examineAll:
                for i in range(self.pr_sample_num):
                    numChanged += self.__examineExample(i)
            else:
                i = self.m_I0.getNext(-1)
                while True:
                    numChanged += self.__examineExample(i)
                    if(self.m_bUp > self.m_bLow - 2 * self.pr_tol):
                        numChanged = 0
                        break
                    i = self.m_I0.getNext(i)
                    if (i == -1):
                        break
            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True
            self.pr_optimization_times += 1

        self.m_b = (self.m_bLow + self.m_bUp) / 2

        if (self.pr_kernal_type == 'linear'):
            self.__get_w_and_b((X,Y))
        self.__prepare_m_f()

        self.NumSVM = 0
        for i in range(self.pr_sample_num):
            if (np.abs(self.m_alph[i] - self.m_alph[i + self.pr_sample_num]) > self.pr_tol):
                self.NumSVM += 1

    def predict(self,X):
        X = self.normalizer.transform(X)
        test_sample_count = X.shape[0]
        X = X.reshape((X.shape[0] * X.shape[1]))

        result = np.empty((test_sample_count),dtype='float32')

        for row in range(test_sample_count):
            for i in range(self.m_feature_num):
                self.pr_vector_b[i] = X[row * self.m_feature_num + i]
            result[row] = self.__f_external(self.pr_vector_b)

        return self.__recovery_Y(result)

if __name__=='__main__':
    df = pd.read_csv('test.csv')

    data = df.values

    X = data[:,1:]
    Y = data[:,0]

    svr = SVR({'kernel_type':'polynomial','epsilon':0.05,'cost':10.0,'max_opt_times':10000,'rbf_gamma':1.0,'coef_lin':1.0,'coef_const':1.0,'poly_degree':2})

    svr.fit(X,Y)
    predicted = svr.predict(X)

    print(np.corrcoef(Y,predicted))


