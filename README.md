# Sign-Language-Translation
This model is a combination between two models. The first one is [Stochastic CSLR](https://github.com/zheniu/stochastic-cslr) which is used to extract glosses from the video frames. Then the [Transformer](https://github.com/kayoyin/transformer-slt) model is used to translate glosses to text.
# Dependencies : 
* Nvidia Cuda Drivers (release 10.1 V10.1.243) to be able to utilize GPU
* Anaconda Virtual Environment
* Python 3.7.6
* Pytorch (version 1.7.0)
* [OpenMT](https://github.com/OpenNMT/OpenNMT-py) library
* Install sclite for evaluating the recognition model alone using WER
* Download [RWTH-PHOENIX-2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) datset
# Testing the Stochastic CSLR model:
1. Clone the stochastic CSLR model from [here](https://github.com/zheniu/stochastic-cslr)
2. Place the folder [ckpt]() inside the /stochastic-cslr-main path
3. Run the following command inside the /stochastic-cslr-main path to test the existing model
```
tzq config/sfl-fp16.yml test
```
Doing these steps will result in the output file hyp.txt which is the corresponding glosses of the test RWTH-PHOENIX-2014 dataset. It should be now fed as an input to the transformer model 
# Testing the Transformer model:
1. Clone the Transformer model code from [here](https://github.com/kayoyin/transformer-slt)
2. Copy the hyp.txt file inside the path /stochastic-cslr-main/stochastic-cslr-main/results/sfl-fp16/30/test an place it in the path /transformer-slt/data
3. Place the folders [model_step_11.pt](), [model_step_12.pt](), and [model_step_13.pt]() inside the path /transformer-slt
4. Run the following command inside /transformer-slt path to do testing while ensembling the weights of the best 3 models
```
python translate.py -model model_step_11.pt model_step_12.pt model_step_13.pt -src data/hyp.tx -output pred.txt -gpu 0 -replace_unk -beam_size 4
```
6. The pred.txt file is the resultant translation. To evaluate it using BlEU scores type the following commands
```
python tools/bleu.py 1 pred.txt data/phoenix2014T.test.de
python tools/bleu.py 2 pred.txt data/phoenix2014T.test.de
python tools/bleu.py 3 pred.txt data/phoenix2014T.test.de
python tools/bleu.py 4 pred.txt data/phoenix2014T.test.de
```

