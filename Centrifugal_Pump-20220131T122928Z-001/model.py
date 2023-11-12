# -*- coding: utf-8 -*-
import numpy as np
from tensorflow import keras

class lenet_kh(keras.Model):
    def __init__(self):
        super(lenet_kh, self).__init__('lenet_kh')
        self.conv_1 = keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            
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
            padding='same',
            activation='relu')
        

        self.pool_2 = keras.layers.MaxPool2D(
           
            pool_size=(2,2),
            strides=(1,1),
            padding='same')

        self.dropout_2 = keras.layers.Dropout(0.25)
        
        

        self.conv_3 = keras.layers.Conv2D(
            filters=128,
            kernel_size=3,
            
            strides=(1,1),
            padding='same',
            activation='relu')
        self.pool_3 = keras.layers.MaxPool2D(
           
            pool_size=(2,2),
            strides=(1,1),
            padding='same')
        
        self.dropout_3 = keras.layers.Dropout(0.25)
        
        
        self.conv_4 = keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            
            strides=(1,1),
            padding='same',
            activation='relu')
        self.pool_4 = keras.layers.MaxPool2D(
           
            pool_size=(2,2),
            strides=(1,1),
            padding='same')
        
        self.dropout_4 = keras.layers.Dropout(0.25)
        
        

        self.flatten = keras.layers.Flatten()

        self.dense_1 = keras.layers.Dense(
            units=2560,
           
            activation='relu')

        

        self.dense_2 = keras.layers.Dense(
            units=5,
            activation='sigmoid')

    def call(self, inputs, training=None, mask=None, **kwargs):
     
        
        temp = self.conv_1(inputs)
        temp = self.pool_1(temp)
        temp = self.dropout_1(temp, training = training)
     
        temp = self.conv_2(temp)
        temp = self.pool_2(temp)
        temp = self.dropout_2(temp, training = training)
        
        
        temp = self.conv_3(temp)
        temp = self.pool_3(temp)
        temp = self.dropout_3(temp, training = training)
        
        temp = self.conv_4(temp)
        temp = self.pool_4(temp)
        temp = self.dropout_4(temp, training = training)
  
       
   
        temp = self.flatten(temp)
        temp = self.dense_1(temp)
        output = self.dense_2(temp)
      

        return output

