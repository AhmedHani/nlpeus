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
from models.torch_charnn import CharRNN
from common.trainer import SupervisedTrainer
from common.transformations import TextTransformations
from utils.text_utils import TextDatasetAnalyzer, TextEncoder, Preprocessor


class StyleRecognizer(object):
    pass


if __name__ == "__main__":
    recognizer = StyleRecognizer()