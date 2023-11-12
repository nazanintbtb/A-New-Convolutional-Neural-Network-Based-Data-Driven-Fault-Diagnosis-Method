# -*- coding: utf-8 -*-
import numpy as np
from tensorflow import keras

class lenet_kh(keras.Model):
    def __init__(self):
        super(lenet_kh, self).__init__('lenet_kh')
        self.conv_1 = keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            
            strides=(1,1),
            padding='same',
            activation='relu')
        

        self.pool_1 = keras.layers.MaxPool2D(
        
            pool_size=(2,2),
            strides=(1,1),
            padding='same')

        self.dropout_1 = keras.layers.Dropout(0.25)

        self.conv_2 = keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            
            strides=(1,1),
            padding='valid',
            activation='relu')
        

        self.pool_2 = keras.layers.MaxPool2D(
       
            pool_size=(2,2),
            strides=(2,2),
            padding='valid')

        self.dropout_2 = keras.layers.Dropout(0.25)

      

        self.flatten = keras.layers.Flatten()

        self.dense_1 = keras.layers.Dense(
            units=8,
           
            activation='relu')

        

        self.dense_2 = keras.layers.Dense(
            units=3,
            activation='sigmoid')

    def call(self, inputs, training=None, mask=None, **kwargs):
     
        # Convolution Layer 1
           
        temp = self.conv_1(inputs)
    

        temp = self.pool_1(temp)
     
        # Dropout Layer 1
        temp = self.dropout_1(temp, training = training)
     
        temp = self.conv_2(temp)
    
        temp = self.pool_2(temp)
 
        temp = self.dropout_2(temp, training = training)

       # temp = self.conv_3(temp)
    
        #temp = self.dropout_3(temp, training = training)
    


        # Flatten Layer 1
 
        temp = self.flatten(temp)
      

        # Fully Connection Layer 1

        temp = self.dense_1(temp)
    
        # Fully Connection Layer 2
  
    
        output = self.dense_2(temp)
      

        return output

