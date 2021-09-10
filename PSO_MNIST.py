#%%
import random
import cv2
import numpy as np
import shutil
import tensorflow as tf
from PIL import Image
from keras import backend as K
from art.estimators.classification import KerasClassifier
from art.metrics import empirical_robustness, loss_sensitivity
import keras.backend as KTF
from deepcoverage import Coverage
from deepgauge_cov import *
from keras.datasets.mnist import load_data
from keras.models import load_model
from keras.utils import to_categorical
from collections import defaultdict
from keras.models import Model
import foolbox
import matplotlib.pyplot as plt
from keras.layers import Input
import time
import math
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
input_tensor = Input(shape=input_shape)
size = 28
shape = (img_rows, img_cols, 1) #mask
shapes = (1, img_cols, img_cols, 1)

#%%
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
def init_coverage_tables(model1):
    model_layer_dict1 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    return model_layer_dict1

def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def coverage(input_data, model, model_layer_dict, threshold=0):
    get_value = []
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            get_value.append(np.mean(scaled[..., num_neuron]))
            # if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
            #     model_layer_dict[(layer_names[i], num_neuron)] = True
    return get_value


def neuron_sensitivity(x_adv, orig_image, model, threshold=0.005):
    # layer_names = [layer.name for layer in model.layers if
    #                'flatten' not in layer.name and 'input' not in layer.name]

    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name and
                   'activation' not in layer.name and 'batch_normalization' not in layer.name
                   and 'dropout' not in layer.name and 'cifa10' not in layer.name and 'before' not in layer.name]

    total_neurons = 0
    sens_neurons = 0
    for i in range(len(layer_names)):
        func = KTF.function(inputs=[model.input], outputs=[model.get_layer(layer_names[i]).output])
        sens = func([x_adv])[0] - func([orig_image])[0]
        scaled = scale(sens[0])
        total_neurons += scaled.shape[-1]
        for num_neuron in range(scaled.shape[-1]):
            tmp_sens = np.mean(scaled[..., num_neuron])
            if tmp_sens > threshold:
                sens_neurons += 1

        # m_sens = sens.flatten()
        # total_neurons += m_sens.shape[0]
        # m_sens[m_sens < threshold] = 0
        # m_sens[m_sens > 0] = 1
        # sens_neurons += np.sum(m_sens)
    return sens_neurons / total_neurons

class PSO():
    def __init__(self, model1, x, k_1, max_iter, gbest_fit, boundary=None, y=None):
        self.w = 0.2
        self.w_end = 2
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.5
        self.r2 = 0.7
        self.Vmax = 0.7
        self.l1 = 0
        self.X = x
        self.pN = x.shape[0]
        self.ori_img = x
        self.y = y
        self.init_table = init_coverage_tables(model1)
        # self.VG = None
        self.seed_useful = []                               
        self.seed_mid = []                                  
        self.particle_useful = []                        
        self.particle_mid = []                            
        self.k_1 = k_1                                      
        self.max_iter = max_iter                         
        self.model = model1
        self.boundary = boundary
   
        self.c_1 = np.resize(self.c1, new_shape=(shape))
        self.c_2 = np.resize(self.c2, new_shape=(shape))
        # print(self.c_1)
        self.V = np.random.uniform(0, self.Vmax, size=(self.pN, size, size, 1))
        self.pbest = np.zeros(shapes)       
        self.gbest = np.zeros(shapes)
        self.p_fit = np.zeros(self.pN)                      
        self.fit = gbest_fit                               
        self.max_object = 2000                           
        self.fits = [self.max_object]
        self.init_Population()                             


 
    def init_Population(self):
     
        # for i in range(self.pN):
   
        #     # self.X[i] = self.noisy_road[i]
        #     # self.V[i] = np.random.uniform(-self.Vmax, self.Vmax, size=(size,size,3))
        self.pbest = self.X
        tmp = self.function(self.X, self.ori_img, self.model, self.y, self.init_table, self.boundary)
        # self.VG = tmp
        self.p_fit = tmp
        print(self.p_fit)
        # index = tmp.argmin()
        bes = tmp
        print("self.fit: ",bes)

        if(bes > self.fit):
            self.fit = bes
            self.gbest = self.X


         
    def iterator(self):
        for t in range(self.max_iter):            
            report = []

            self.w = self.w_end + (self.w - self.w_end) * (self.max_iter - t) / self.max_iter
            # w = np.resize(self.w,new_shape=(shape))
            # r1,r2
            r_1 = self.r1
            r_2 = self.r2
            if len(self.X.shape) == 4:
                temp = self.function(self.X, self.ori_img, self.model, self.y, self.init_table, self.boundary)
            else:
                temp = self.function(np.expand_dims(self.X, axis=0), self.ori_img, self.model, self.init_table, self.boundary)

            if temp > self.p_fit:
                self.p_fit = temp
                self.pbest = self.X
                # print(self.pbest[i].shape)
                if temp > self.fit:
                    self.fit = temp
                    self.gbest = self.pbest
                    # cv2.imwrite("./hh.png", self.gbest.reshape(32, 32, 3))
            self.V = self.V * self.w + self.c_1 * np.random.uniform(0, 1) * (self.pbest - self.X) + self.c_2 * np.random.uniform(0, 1) * (self.gbest - self.X)
            self.X = self.X + self.V
           
            self.V = np.clip(self.V, -self.Vmax, self.Vmax)
            self.X = np.clip(self.X, 0, 1)
            print(self.fit)
        return self.fit, self.gbest

      

    def function(self, inputs, ori_image, model, y, model_layer_dict1, boundary):
        update_coverage(inputs, models, model_layer_dict1, 0)
        neurons = len(model_layer_dict1)
        neuron_value = coverage(inputs, models, model_layer_dict1, 0)
        nbc, snac = nbcov_and_snacov_one(neuron_value, neurons, boundary)
        classifier = KerasClassifier(clip_values=(0, 1), model=model, preprocessing=(0, 1))
        ss = neuron_sensitivity(inputs, ori_image, model, threshold=0.2)
        se = loss_sensitivity(classifier, ori_image, y)
        prediction_func = KTF.function(inputs=[model.input], outputs=[model.layers[-1].output])
        loss = - prediction_func([inputs])[0][:, np.argmax(models.predict(inputs)+1)]
        loss, _ = model.evaluate(inputs, y)
        # return neuron_covered(model_layer_dict1)[2]
        return nbc + ss
#%%
(x_train, y_train), (x_test, y_test) = load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
x_train = x_train / 255.0
x_test = x_test / 255.0
model = load_model('./Model3.h5')
generation = []
time_start = time.time()
for index in range(100):
    test = PSO(model, x_test[index:index+1], k_1=0.5, max_iter=20, gbest_fit=-5)
    gbest_fit, gbest = test.iterator()
    generation.append(gbest)
    # pso_values.append(coverage(gbest, model, model_layer_dict1))
time_end = time.time()
