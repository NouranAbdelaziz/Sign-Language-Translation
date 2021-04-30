# Sign-Language-Translation
This model is a combination between two models. The first one is [Stochastic CSLR](https://github.com/zheniu/stochastic-cslr) which is used to extract glosses from the video frames. Then the [Transformer](https://github.com/kayoyin/transformer-slt) model is used to translate glosses to text.
# Dependencies : 
* Nvidia Cuda Drivers (release 10.1 V10.1.243) to be able to utilize GPU
* Anaconda Virtual Environment
* Python 3.7.6
* Pytorch (version 1.7.0)
* [OpenMT](https://github.com/OpenNMT/OpenNMT-py) library
* Install sclite for evaluating the recognition model alone using WER
* Install [RWTH-PHOENIX-2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) datset
# Testing the Stochastic CSLR model:
1. Clone the stochastic CSLR model from [here](https://github.com/zheniu/stochastic-cslr)
2. Place the folder [ckpt]() inside the /stochastic-cslr-main path
3. Run the following command to test the existing model
|tzq config/sfl-fp16.yml test|

