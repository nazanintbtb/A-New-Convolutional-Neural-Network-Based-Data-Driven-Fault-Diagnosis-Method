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

#------------------------------------------------------
# extract normal signal to create data set
def data_train_test():
  normal_train_datas=[]
  normal_train_labels=[]
  normal_test_datas=[]
  normal_test_labels=[]
  normal_valid_datas=[]
  normal_valid_labels=[]

  for k in range(1,6,1):
    ff='./normal/normal%d.TXT'%k
  
    with open(ff) as f:
      lines = f.readlines()
    randomdig=[]
 
    rand=random.randint(0,16383 )
    for i in range(2420):
      randomdig.append(rand)
      inc=rand
      data=[]
  
      for j in range(4096):
    
        data.append(float(lines[inc]))
        inc=inc+1

      rand=random.randint(0,16383 )
      while((rand in randomdig)):
        rand=random.randint(0,16383  )
      if(i<2000):
        normal_train_datas.append(data)
        normal_train_labels.append([1,0,0,0,0]) #create label of signals
      elif(2000<=i<2400):
        normal_test_datas.append(data)
        normal_test_labels.append([1,0,0,0,0])
      else:
        normal_valid_datas.append(data)
        normal_valid_labels.append([1,0,0,0,0])
      
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
# extract impeller_wearing signal to create data set

  iw_train_datas=[]
  iw_test_datas=[]
  iw_valid_datas=[]
  iw_train_labels=[]
  iw_test_labels=[]
  iw_valid_labels=[]
  for k in range(1,6,1):
    ff='./impeller_wearing/iw%d.TXT'%k
  
    with open(ff) as f:
      lines = f.readlines()
    randomdig=[]
 
    rand=random.randint(0,16383 )
    for i in range(2420):
      randomdig.append(rand)
      inc=rand
      data=[]
  
      for j in range(4096):
    
        data.append(float(lines[inc]))
        inc=inc+1

      rand=random.randint(0,16383 )
      while((rand in randomdig)):
        rand=random.randint(0,16383 )
      if(i<2000):
        iw_train_datas.append(data)
        iw_train_labels.append([0,1,0,0,0]) #create label of signals

      elif(2000<=i<2400):
        iw_test_datas.append(data)
        iw_test_labels.append([0,1,0,0,0])
      else:
        iw_valid_datas.append(data)
        iw_valid_labels.append([0,1,0,0,0])

  iw_train_datas=np.array(iw_train_datas)
  iw_test_datas=np.array(iw_test_datas)
  iw_valid_datas=np.array(iw_valid_datas)
  iw_train_labels=np.array(iw_train_labels)
  iw_test_labels=np.array(iw_test_labels)
  iw_valid_labels=np.array(iw_valid_labels)
 
 # print(ps_train_datas.shape)
 # print(ps_test_datas.shape)
 # print(ps_train_labels.shape)
 # print(ps_test_labels.shape)

#--------------------------------------
# extract bearing roller wearing signal to create data set

  br_train_datas=[]
  br_test_datas=[]
  br_valid_datas=[]
  br_train_labels=[]
  br_test_labels=[]
  br_valid_labels=[]
  for k in range(1,6,1):
    ff='./bearing_roller/br%d.TXT'%k
  
    with open(ff) as f:
      lines = f.readlines()
    randomdig=[]
 
    rand=random.randint(0,16383  )
    for i in range(2420):
      randomdig.append(rand)
      inc=rand
      data=[]
  
      for j in range(4096):
    
        data.append(float(lines[inc]))
        inc=inc+1

      rand=random.randint(0,16383  )
      while((rand in randomdig)):
        rand=random.randint(0,16383  )
      if(i<2000):
        br_train_datas.append(data)
        br_train_labels.append([0,0,1,0,0]) #create label of data
      elif(2000<=i<2400):
        br_test_datas.append(data)
        br_test_labels.append([0,0,1,0,0])
      else:
        br_valid_datas.append(data)
        br_valid_labels.append([0,0,1,0,0])
  br_train_datas=np.array(br_train_datas)
  br_test_datas=np.array(br_test_datas)
  br_valid_datas=np.array(br_valid_datas)
  br_train_labels=np.array(br_train_labels)
  br_test_labels=np.array(br_test_labels)
  br_valid_labels=np.array(br_valid_labels)

  #print(vp_train_datas.shape)
  #print(vp_test_datas.shape)
  #print(vp_train_labels.shape)
  #print(vp_test_labels.shape)
  
#----------------------------------------------------------------------------------
# extract outer_race wearing signal to create data set
  or_train_datas=[]
  or_test_datas=[]
  or_valid_datas=[]
  or_train_labels=[]
  or_test_labels=[]
  or_valid_labels=[]
  for k in range(1,6,1):
    ff='./outer_race/or%d.TXT'%k
  
    with open(ff) as f:
      lines = f.readlines()
    randomdig=[]
 
    rand=random.randint(0,16383  )
    for i in range(2420):
      randomdig.append(rand)
      inc=rand
      data=[]
  
      for j in range(4096):
    
        data.append(float(lines[inc]))
        inc=inc+1

      rand=random.randint(0,16383  )
      while((rand in randomdig)):
        rand=random.randint(0,16383  )
      if(i<2000):
        or_train_datas.append(data)
        or_train_labels.append([0,0,0,1,0])
      elif(2000<=i<2400):
        or_test_datas.append(data)
        or_test_labels.append([0,0,0,1,0])
      else:
        or_valid_datas.append(data)
        or_valid_labels.append([0,0,0,1,0])
  or_train_datas=np.array(or_train_datas)
  or_test_datas=np.array(or_test_datas)
  or_valid_datas=np.array(or_valid_datas)
  or_train_labels=np.array(or_train_labels)
  or_test_labels=np.array(or_test_labels)
  or_valid_labels=np.array(or_valid_labels)
#--------------------------------------------------------------------------------
# extract inner_race wearing signal to create data set

  ir_train_datas=[]
  ir_test_datas=[]
  ir_valid_datas=[]
  ir_train_labels=[]
  ir_test_labels=[]
  ir_valid_labels=[]
  for k in range(1,6,1):
    ff='./inner_race/ir%d.TXT'%k
  
    with open(ff) as f:
      lines = f.readlines()
    randomdig=[]
 
    rand=random.randint(0,16383  )
    for i in range(2420):
      randomdig.append(rand)
      inc=rand
      data=[]
  
      for j in range(4096):
    
        data.append(float(lines[inc]))
        inc=inc+1

      rand=random.randint(0,16383  )
      while((rand in randomdig)):
        rand=random.randint(0,16383  )
      if(i<2000):
        ir_train_datas.append(data)
        ir_train_labels.append([0,0,0,0,1]) #create label of dataset
      elif(2000<=i<2400):
        ir_test_datas.append(data)
        ir_test_labels.append([0,0,0,0,1])
      else:
        ir_valid_datas.append(data)
        ir_valid_labels.append([0,0,0,0,1])
  ir_train_datas=np.array(ir_train_datas)
  ir_test_datas=np.array(ir_test_datas)
  ir_valid_datas=np.array(ir_valid_datas)
  ir_train_labels=np.array(ir_train_labels)
  ir_test_labels=np.array(ir_test_labels)
  ir_valid_labels=np.array(ir_valid_labels)

#---------------------------------------------------------------------------------------
#convert array that contain signals to image 
  normal_train_img=convert_to_image(normal_train_datas)
  normal_test_img=convert_to_image(normal_test_datas)
  normal_valid_img=convert_to_image(normal_valid_datas)

  iw_train_img=convert_to_image(iw_train_datas)
  iw_test_img=convert_to_image(iw_test_datas)
  iw_valid_img=convert_to_image(iw_valid_datas)

  br_train_img=convert_to_image(br_train_datas)
  br_test_img=convert_to_image(br_test_datas)
  br_valid_img=convert_to_image(br_valid_datas)
  
  or_train_img=convert_to_image(or_train_datas)
  or_test_img=convert_to_image(or_test_datas)
  or_valid_img=convert_to_image(or_valid_datas)
  
  ir_train_img=convert_to_image(ir_train_datas)
  ir_test_img=convert_to_image(ir_test_datas)
  ir_valid_img=convert_to_image(ir_valid_datas)

 # create train and valid and test dataset  
  trains_data=normal_train_img + iw_train_img + br_train_img + or_train_img + ir_train_img
  tests_data=normal_test_img + iw_test_img + br_test_img + or_test_img + ir_test_img
  valids_data=normal_valid_img + iw_valid_img + br_valid_img + or_valid_img + ir_valid_img

  d = np.concatenate((normal_valid_labels,iw_valid_labels),axis=0)
  e=np.concatenate((d,br_valid_labels),axis=0)
  f=np.concatenate((e,or_valid_labels),axis=0)
  valids_labels = np.concatenate((f,ir_valid_labels),axis=0)

  d = np.concatenate((normal_train_labels,iw_train_labels),axis=0)
  e=np.concatenate((d,br_train_labels),axis=0)
  f=np.concatenate((e,or_train_labels),axis=0)
  trains_labels = np.concatenate((f,ir_train_labels),axis=0)

  d = np.concatenate((normal_test_labels,iw_test_labels),axis=0)
  e=np.concatenate((d,br_test_labels),axis=0)
  f=np.concatenate((e,or_test_labels),axis=0)
  tests_labels = np.concatenate((f,ir_test_labels),axis=0)
 
  
  
  return trains_data,tests_data,valids_data,trains_labels,tests_labels,valids_labels
 
#-----------------------------------------------------------------------
# same methode of article to conver signal to image
def convert_to_image(all_arr):
    
  img_array=[]
  for i in range(all_arr.shape[0]):
    arr=all_arr[i]
    min=np.amin(arr)
    max=np.amax(arr)
    P = np.zeros((64,64), dtype=np.uint8)
    for j in range(1,65,1):

      for k in range(1,65,1):
        index=((j-1)*64)+k
        surat=arr[index-1]-min
        makhraj=max-min
        amoant=(surat/makhraj)*255
        P[(j-1)][(k-1)]=amoant     # amount of pixel in 0 to 255 range to create grayscale image


    img = Image.fromarray(P)
    img_array.append(img)
  return img_array

#----------------------------------------------------------------------------------------
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) 
#--------------------------------------------------------------------------------------------------
#serialise data to prepare convert and write them in tfrecord format
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
 
#-------------------------------------------------------------------------------------------
#write trian valid test image dataset to tfrecord file
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