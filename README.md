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
2. Copy the hyp.txt file inside the path /stochastic-cslr-main/stochastic-cslr-main/results/sfl-fp16/30/test and place it in the path /transformer-slt/data
3. Place the folders [model_step_11.pt](), [model_step_12.pt](), and [model_step_13.pt]() inside the path /transformer-slt
4. Run the following command inside /transformer-slt path to do testing while ensembling the weights of the best 3 models
```
python translate.py -model model_step_11.pt model_step_12.pt model_step_13.pt -src data/hyp.txt -output pred.txt -gpu 0 -replace_unk -beam_size 4
```
6. The pred.txt file is the resultant translation. To evaluate it using BlEU scores type the following commands
```
python tools/bleu.py 1 pred.txt data/phoenix2014T.test.de
python tools/bleu.py 2 pred.txt data/phoenix2014T.test.de
python tools/bleu.py 3 pred.txt data/phoenix2014T.test.de
python tools/bleu.py 4 pred.txt data/phoenix2014T.test.de
```
# Training the Stochastic CSLR model:
1. Make sure you are inside the path /stochastic-cslr-main
2. Change the hyperparameters in the dfl-fp16.yml file
3. Train the model using the following command
```
tzq config/sfl-fp16.yml train
```
Note: The training of the model on a high functioning GPU could take about 20 hours for 30 epochs. 
# Training the Transformer model:
1. Install dependencies and make sure you are inside the virtual environment and /transformer-slt path using the following commands
```
# create a new virtual environment
virtualenv --python=python3 venv
source venv/bin/activate

# clone the repo
git clone https://github.com/kayoyin/transformer-slt.git
cd transformer-slt

# install python dependencies
pip install -r requirements.txt

# install OpenNMT-py
python setup.py install
```
3. Preprocess the data using OpenNMT using the following command
```
onmt_preprocess -train_src data/phoenix2014T.train.gloss -train_tgt data/phoenix2014T.train.de -valid_src data/phoenix2014T.dev.gloss -valid_tgt data/phoenix2014T.dev.de -save_data data/dgs -lower 
```
5. Train the model using the tuned hyperparameters using the following command
```
python  train.py -data data/dgs -save_model model -keep_checkpoint 1 \
          -layers 2 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
          -encoder_type transformer -decoder_type transformer -position_encoding \
          -max_generator_batches 2 -dropout 0.1 \
          -early_stopping 3 -early_stopping_criteria accuracy ppl \
          -batch_size 2048 -accum_count 3 -batch_type tokens -normalization tokens \
          -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 3000 -learning_rate 0.5 \
          -max_grad_norm 0 -param_init 0  -param_init_glorot \
          -label_smoothing 0.1 -valid_steps 100 -save_checkpoint_steps 100 \
          -world_size 1 -gpu_ranks 0
```
For testing the new trained model, repeat the steps of the testing above but using the new models obtained. 
