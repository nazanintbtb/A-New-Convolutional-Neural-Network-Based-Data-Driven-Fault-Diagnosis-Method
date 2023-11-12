# A-New-Convolutional-Neural-Network-Based-Data-Driven-Fault-Diagnosis-Method

In this paper, a convolutional neural network-based learning model is used for image classification. It aims to detect errors in three data groups, including motor bearing data, centrifugal pump data, and axial piston hydraulic pump data.
https://ieeexplore.ieee.org/abstract/document/8114247
# Main idea
The idea of this paper is to convert raw signals into images. The workflow is as follows: in order to obtain an image of size M*M, the signal needs to be randomly divided into M segments, each consisting of M samples of the signal. Based on this, we will have a total of M^2 samples, and we fill the pixels of the M*M image according to the image number (1) specified in the paper. The formula for calculating each pixel of the image is provided in the paper. Then the preprocessed inputs are fed into a learning model with a 4-layer convolutional neural network architecture and 4 layers of 4D pooling after each convolutional layer.
![modell](https://github.com/nazanintbtb/A-New-Convolutional-Neural-Network-Based-Data-Driven-Fault-Diagnosis-Method/assets/88847995/9603d277-00af-4c49-aee4-4dac4e61b624)
# preproccess output
![22](https://github.com/nazanintbtb/A-New-Convolutional-Neural-Network-Based-Data-Driven-Fault-Diagnosis-Method/assets/88847995/dc0ff649-72d4-48c6-84fd-12f51440e179)
![Screenshot (71)](https://github.com/nazanintbtb/A-New-Convolutional-Neural-Network-Based-Data-Driven-Fault-Diagnosis-Method/assets/88847995/69f97e3c-7352-4235-8498-6e775cd78c23)
![33](https://github.com/nazanintbtb/A-New-Convolutional-Neural-Network-Based-Data-Driven-Fault-Diagnosis-Method/assets/88847995/4f16dee5-6181-4ca5-9a4c-e906879e953c)




