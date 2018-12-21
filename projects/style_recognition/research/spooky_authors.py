# Copyright (c) 2018-present, Ahmed H. Al-Ghidani.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__author__ = "Ahmed H. Al-Ghidani"
__copyright__ = "Copyright 2018, The nlpeus Project, https://github.com/AhmedHani/nlpeus"
__license__ = "GPL"
__maintainer__ = "Ahmed H. Al-Ghidani"
__email__ = "ahmed.hani.ibrahim@gmail.com"

import argparse

from common.batcher import Batcher
from models.torch_charnn import CharRNN
from common.trainer import SupervisedTrainer
from common.transformations import TextTransformations
from utils.text_utils import TextDatasetAnalyzer, TextEncoder, Preprocessor
from common.experiment import SupervisedExperiment, SupervisedExperimentSummarizer
from projects.style_recognition.research.data_processing import SpookyAuthorsDataProcessing


parser = argparse.ArgumentParser(description='Style Recognition training playground')

parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
parser.add_argument('--epochs', type=int, default=3, help='number of training epochs')
parser.add_argument('--max-charslen', type=int, default=50, help='max chars length that will be fed to the network')
parser.add_argument('--max-wordslen', type=int, default=10, help='max words length that will be fed to the network')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--min-wordsfreq', type=int, default=10, help='min words frequency to be considered')
parser.add_argument('--min-charsfreq', type=int, default=100, help='min chars frequency to be considered')

args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
max_charslen = args.max_charslen
max_wordslen = args.max_wordslen
min_wordsfreq = args.min_wordsfreq
min_charsfreq = args.min_charsfreq
device = 'cpu' if not args.no_cuda is False else 'cuda'

dp = SpookyAuthorsDataProcessing(train_file='./projects/style_recognition/datasets/spooky_authors/train.csv',
                                 test_file='./projects/style_recognition/datasets/spooky_authors/test.csv',
                                 preprocessing=False)

class2index, index2class = dp.class2index, dp.index2class

dataset_analyzer = TextDatasetAnalyzer(data=dp.train_data, data_axis={'text': 1, 'label': 2},
                                       index2class=index2class,
                                       outpath='./style_recognition/datasets/spooky_authors/train_analysis.log')

dataset_analyzer.all()
del dataset_analyzer