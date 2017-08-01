# VQS

Source code for VQS: Linking Segmentations to Questions and Answers for Supervised Attention in VQA and Question-Focused Semantic Segmentation. This current code can get 69.8 on Multiple-Choice task on test-standard split of [VQA v1](http://www.visualqa.org/vqa_v1_challenge.html). 

## Requirements

* This code requires [caffe](http://caffe.berkeleyvision.org/). The preprocssing code is in Python, and you need to install [NLTK](http://www.nltk.org/) if you want to use NLTK to tokenize the question.
* You need to install [gensim](https://radimrehurek.com/gensim/install.html) and download the pretrained [word2vec](https://code.google.com/archive/p/word2vec) (This is `model.bin` in `processJson.py` and `writeSentenceMat.py`).

## Download Dataset

* [VQS_data](https://www.dropbox.com/sh/i9cucdn8ronfytl/AAA8asyE4j91knyinYygJPapa?dl=0)
* [VQA_data](http://www.visualqa.org/vqa_v1_download.html)

## Preprocess data

You need to download `VQA_data` and unzip them into `./mlp` folder.

Firstly you need to extract the penultimate layer of Resnet-101 to represent image and write the image feature into `feaTrainPool5.txt` and `feaValPool5.txt` with the order in 'trainList.txt' and 'valList.txt'

Then
```
python processJson.py
python normVec.py
```
to get concatenate l2 normalized quesion and answer feature

And then

```
python writelmdb.py
```
to concatenate the l2 normalized image feature, question feature and answer feature into LMDB to feed into neural network (MLP).

### MLP

This code implement a strong baseline from Facebook: [Revisiting Visual Question Answering Baselines](https://arxiv.org/pdf/1606.08390.pdf)

### Supervised attention

This code implement a method similar to [Stacked attention networks for image question answering](https://arxiv.org/abs/1511.02274)

Firstly you need to download `VQS_data` and unzip them into `./supervise_attention` folder.
Then you need to extract the 'res5c' layer of Resnet-101 to represent image. (Extracted the features from 448x448 image)

```
python getAttentLabel.py
python writeSentenceMat.py
```
To get label and question feature LMDB to feed into neural network.

And you can get `attention feature` with this model.

### Train MLP with `attention feature`

Now you can concat `attention feature` into MLP model

These processes are a little complicated, please feel free to ask me if you have some questions.

