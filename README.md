# AlexNet_3dConv
TensorFlow implementation of AlexNet(2012) by Alex Krizhevsky, with 3D convolutiional layers.
### 3D AlexNet
![3D AlexNet](https://github.com/denti/AlexNet_3dConv/blob/master/img/AlexNet_3D.jpeg)
Network with a standart AlexNet architecture, but it has 3D instead 2D filters on each `Conv` and `Pool` layers. 
### Standart AlexNet
![Standart AlexNet](https://github.com/denti/AlexNet_3dConv/blob/master/img/AlexNet.jpeg)



### To fit  it can be implemented with @OpenAI's realization of 'Fitting larger networks into memory' [https://github.com/openai/gradient-checkpointing](https://github.com/openai/gradient-checkpointing)
* Note: This model needs a lot of GPU memory. Training session will not starts on 1 GPU without additional data separation or code's parallelization (or https://github.com/openai/gradient-checkpointing for example). 

* Note2: This model doesn't pretend to be the `SilverBullet` in 3D image recognition (like AlexNet was in 2D). It's just an example of 3D convolutional model in TensorFlow.
