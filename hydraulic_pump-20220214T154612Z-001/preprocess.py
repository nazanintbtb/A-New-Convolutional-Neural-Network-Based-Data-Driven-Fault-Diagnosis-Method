import random
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import tensorflow as tf
import scipy.io as sio

from tqdm import tqdm


def data_train_test():
  normal_train_datas=[]
  normal_train_labels=[]
  normal_test_datas=[]
  normal_test_labels=[]
  normal_valid_datas=[]
  normal_valid_labels=[]

  for k in range(1,4,1):
    ff='./normal/normal%d.txt'%k
  
    with open(ff) as f:
      lines = f.readlines()
    randomdig=[]
 
    rand=random.randint(0,767 )
    for i in range(520):
      randomdig.append(rand)
      inc=rand
      data=[]
  
      for j in range(256):
    
        data.append(float(lines[inc]))
        inc=inc+1

      rand=random.randint(0,767 )
      while((rand in randomdig)):
        rand=random.randint(0,767 )
      if(i<400):
        normal_train_datas.append(data)
        normal_train_labels.append([1,0,0])
      elif(400<=i<500):
        normal_test_datas.append(data)
        normal_test_labels.append([1,0,0])
      else:
        normal_valid_datas.append(data)
        normal_valid_labels.append([1,0,0])
      
  normal_train_datas=np.array(normal_train_datas)
  normal_test_datas=np.array(normal_test_datas)
  normal_valid_datas=np.array(normal_valid_datas)

  normal_train_labels=np.array(normal_train_labels)
  normal_test_labels=np.array(normal_test_labels)
  normal_valid_labels=np.array(normal_valid_labels)
 

  #print(normal_train_datas.shape)
  #print(normal_test_datas.shape)
  #print(normal_train_labels.shape)
  #print(normal_test_labels.shape)

#----------------------------------------------------
#piston_shoes wearing
  ps_train_datas=[]
  ps_test_datas=[]
  ps_valid_datas=[]
  ps_train_labels=[]
  ps_test_labels=[]
  ps_valid_labels=[]
  for k in range(1,7,1):
    ff='./piston_shoes/ps%d.txt'%k
  
    with open(ff) as f:
      lines = f.readlines()
    randomdig=[]
 
    rand=random.randint(0,767 )
    for i in range(520):
      randomdig.append(rand)
      inc=rand
      data=[]
  
      for j in range(256):
    
        data.append(float(lines[inc]))
        inc=inc+1

      rand=random.randint(0,767 )
      while((rand in randomdig)):
        rand=random.randint(0,767 )
      if(i<400):
        ps_train_datas.append(data)
        ps_train_labels.append([0,1,0])

      elif(400<=i<500):
        ps_test_datas.append(data)
        ps_test_labels.append([0,1,0])
      else:
        ps_valid_datas.append(data)
        ps_valid_labels.append([0,1,0])

  ps_train_datas=np.array(ps_train_datas)
  ps_test_datas=np.array(ps_test_datas)
  ps_valid_datas=np.array(ps_valid_datas)
  ps_train_labels=np.array(ps_train_labels)
  ps_test_labels=np.array(ps_test_labels)
  ps_valid_labels=np.array(ps_valid_labels)
 
 # print(ps_train_datas.shape)
 # print(ps_test_datas.shape)
 # print(ps_train_labels.shape)
 # print(ps_test_labels.shape)

#--------------------------------------
#valvee plate wearing
  vp_train_datas=[]
  vp_test_datas=[]
  vp_valid_datas=[]
  vp_train_labels=[]
  vp_test_labels=[]
  vp_valid_labels=[]
  for k in range(1,5,1):
    ff='./valve_plate/vp%d.txt'%k
  
    with open(ff) as f:
      lines = f.readlines()
    randomdig=[]
 
    rand=random.randint(0,767 )
    for i in range(520):
      randomdig.append(rand)
      inc=rand
      data=[]
  
      for j in range(256):
    
        data.append(float(lines[inc]))
        inc=inc+1

      rand=random.randint(0,767 )
      while((rand in randomdig)):
        rand=random.randint(0,767 )
      if(i<400):
        vp_train_datas.append(data)
        vp_train_labels.append([0,0,1])
      elif(400<=i<500):
        vp_test_datas.append(data)
        vp_test_labels.append([0,0,1])
      else:
        vp_valid_datas.append(data)
        vp_valid_labels.append([0,0,1])
  vp_train_datas=np.array(vp_train_datas)
  vp_test_datas=np.array(vp_test_datas)
  vp_valid_datas=np.array(vp_valid_datas)
  vp_train_labels=np.array(vp_train_labels)
  vp_test_labels=np.array(vp_test_labels)
  vp_valid_labels=np.array(vp_valid_labels)

  #print(vp_train_datas.shape)
  #print(vp_test_datas.shape)
  #print(vp_train_labels.shape)
  #print(vp_test_labels.shape)


  normal_train_img=convert_to_image(normal_train_datas)
  normal_test_img=convert_to_image(normal_test_datas)
  normal_valid_img=convert_to_image(normal_valid_datas)

  ps_train_img=convert_to_image(ps_train_datas)
  ps_test_img=convert_to_image(ps_test_datas)
  ps_valid_img=convert_to_image(ps_valid_datas)

  vp_train_img=convert_to_image(vp_train_datas)
  vp_test_img=convert_to_image(vp_test_datas)
  vp_valid_img=convert_to_image(vp_valid_datas)

   
  trains_data=normal_train_img + ps_train_img + vp_train_img
  tests_data=normal_test_img + ps_test_img + vp_test_img
  valids_data=normal_valid_img + ps_valid_img + vp_valid_img

  d = np.concatenate((normal_valid_labels,ps_valid_labels),axis=0)
  valids_labels = np.concatenate((d,vp_valid_labels),axis=0)

  d = np.concatenate((normal_train_labels,ps_train_labels),axis=0)
  trains_labels = np.concatenate((d,vp_train_labels),axis=0)

  d = np.concatenate((normal_test_labels,ps_test_labels),axis=0)
  tests_labels = np.concatenate((d,vp_test_labels),axis=0)
 
  
  
  return trains_data,tests_data,valids_data,trains_labels,tests_labels,valids_labels
 
 
def convert_to_image(all_arr):
    
  img_array=[]
  for i in range(all_arr.shape[0]):
    arr=all_arr[i]
    min=np.amin(arr)
    max=np.amax(arr)
    P = np.zeros((16,16), dtype=np.uint8)
    for j in range(1,17,1):

      for k in range(1,17,1):
        index=((j-1)*16)+k
        surat=arr[index-1]-min
        makhraj=max-min
        amoant=(surat/makhraj)*255
        P[(j-1)][(k-1)]=amoant


    img = Image.fromarray(P)
    img_array.append(img)
  return img_array


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) 


def serialize_example(image, label, image_shape):
   feature = {
        'image': _bytes_feature(image),
      
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label.flatten())),
    }
 
#  Create a Features message using tf.train.Example.
   example_proto =  tf.train.Example(features=tf.train.Features(feature=feature))
   return example_proto.SerializeToString()


def write_to_tfrecords():
  trains_data,tests_data,valids_data,trains_labels,tests_labels,valids_labels=data_train_test()

  with tf.io.TFRecordWriter("./traindata.tfrecords") as writer:
          for i in tqdm(range(len(trains_data)), desc="Processing train Data", ascii=True):
             img=trains_data[i]
             img_array = tf.keras.preprocessing.image.img_to_array(img)
             img_bytes = tf.io.serialize_tensor(img_array)
             image_shape = img_array.shape
             example = serialize_example(img_bytes, trains_labels[i], image_shape)
             writer.write(example)

  with tf.io.TFRecordWriter("./testdata.tfrecords") as writer:
          for i in tqdm(range(len(tests_data)), desc="Processing test Data", ascii=True):
             img=tests_data[i]
             img_array = tf.keras.preprocessing.image.img_to_array(img)
             img_bytes = tf.io.serialize_tensor(img_array)
             image_shape = img_array.shape
             example = serialize_example(img_bytes, tests_labels[i], image_shape)
             writer.write(example)

  with tf.io.TFRecordWriter("./validdata.tfrecords") as writer:
          for i in tqdm(range(len(valids_data)), desc="Processing valid Data", ascii=True):
             img=valids_data[i]
             img_array = tf.keras.preprocessing.image.img_to_array(img)
             img_bytes = tf.io.serialize_tensor(img_array)
             image_shape = img_array.shape
             example = serialize_example(img_bytes, valids_labels[i], image_shape)
             writer.write(example)
  

if __name__ == '__main__':
    # Write the train data and test data to .tfrecord file.
    write_to_tfrecords()
    
    
    print("pre tamam")