# BiDAF-No-Answer
BiDAF model with no-answer prediction, based on [galsang/BiDAF-pytorch](https://github.com/galsang/BiDAF-pytorch) (Re-implementation of BiDAF model)

## Original paper

[BiDAF](https://arxiv.org/abs/1611.01603)(Bidirectional Attention Flow for Machine Comprehension, Minjoon Seo et al., ICLR 2017)

## Results

Dataset: [SQuAD v2.0](https://rajpurkar.github.io/SQuAD-explorer/)

| Model(Single) | EM(%)(dev) | F1(%)(dev) |
|--------------|:----------:|:----------:|
| **Re-implementation** | **57.450** | **61.338** |
| baseline(from paper [SQuAD2.0](http://arxiv.org/abs/1806.03822)) | 59.8 | 62.6 |

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- GPU: Nvidia GTX 1080ti
- Language: Python 3.6.7
- Pytorch: **1.0.1**

## Requirements

Please install the following library requirements specified in the **requirements.txt** first.

    torch==1.0.1
    nltk==3.2.4
    tensorboardX==0.8
    torchtext==0.3.1

## Execution

> python run.py --help

	usage: run.py [-h] [--char-dim CHAR_DIM]
	              [--char-channel-width CHAR_CHANNEL_WIDTH]
	              [--char-channel-size CHAR_CHANNEL_SIZE]
	              [--context-threshold CONTEXT_THRESHOLD]
	              [--dev-batch-size DEV_BATCH_SIZE] [--dev-file DEV_FILE]
	              [--dropout DROPOUT] [--epoch EPOCH]
	              [--exp-decay-rate EXP_DECAY_RATE] [--gpu GPU]
	              [--hidden-size HIDDEN_SIZE] [--learning-rate LEARNING_RATE]
	              [--print-freq PRINT_FREQ] [--train-batch-size TRAIN_BATCH_SIZE]
	              [--train-file TRAIN_FILE] [--word-dim WORD_DIM]
	              [--prediction_file PREDICTION_FILE] [--id ID]
	              [--random_seed RANDOM_SEED]
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --char-dim CHAR_DIM
	  --char-channel-width CHAR_CHANNEL_WIDTH
	  --char-channel-size CHAR_CHANNEL_SIZE
	  --context-threshold CONTEXT_THRESHOLD
	  --dev-batch-size DEV_BATCH_SIZE
	  --dev-file DEV_FILE
	  --dropout DROPOUT
	  --epoch EPOCH
	  --exp-decay-rate EXP_DECAY_RATE
	  --gpu GPU
	  --hidden-size HIDDEN_SIZE
	  --learning-rate LEARNING_RATE
	  --print-freq PRINT_FREQ
	  --train-batch-size TRAIN_BATCH_SIZE
	  --train-file TRAIN_FILE
	  --word-dim WORD_DIM
	  --prediction_file PREDICTION_FILE
	  --id ID
	  --random_seed RANDOM_SEED

