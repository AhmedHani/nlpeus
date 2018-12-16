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

import os
import argparse
import pickle as pkl
from models.torch_charnn import CharRNN
from common.trainer import SupervisedTrainer
from common.transformations import TextTransformations
from utils.text_utils import TextDatasetAnalyzer, TextEncoder, Preprocessor


class StyleRecognizer(object):

    def __init__(self, production_dir='./projects/style_recognition/shared/production'):
        saved_data = os.path.join(production_dir, 'saved_data')
        saved_model = os.path.join(production_dir, 'saved_model')
        saved_pipeline = os.path.join(production_dir, 'saved_pipeline')

        self.__validate_dir_paths(saved_data, saved_model, saved_pipeline)

        model_weights_path = os.path.join(saved_model, 'model_weights.pt')
        trainer_path = os.path.join(saved_pipeline, 'trainer.pkl')

        self.__validate_dir_paths(model_weights_path, trainer_path)

        self.trainer = self.__load_trainer(trainer_path)
        self.trainer = self.__load_model_weights(self.trainer, model_weights_path)

        transformations_path = os.path.join(saved_pipeline, 'transformations.pkl')
        self.transformations = self.__load_transformations(transformations_path)

        encoder_path = os.path.join(saved_pipeline, 'encoder.pkl')
        self.encoder = self.__load_encoder(encoder_path)

        ## show trainer summary
    
    def recognize(self, text):
        if self.transformations is not None and isinstance(self.transformations, list):
            for transformation in self.transformations:
                text = transformation(text)
        
        if self.encoder is not None:
            text = self.encoder.encode(text)

        predictions = self.trainer.predict_classes(text)

        return predictions

    @staticmethod
    def __load_transformations(transformations_path):
        if not os.path.exists(transformations_path):
            return None

        with open(transformations_path, 'rb') as reader:
            return pkl.load(reader)

    @staticmethod
    def __load_encoder(encoder_path):
        if not os.path.exists(encoder_path):
            return None
            
        with open(encoder_path, 'rb') as reader:
            return pkl.load(reader)

    @staticmethod
    def __load_model_weights(trainer, weights_path):
        trainer.load(weights_path)

        return trainer

    @staticmethod
    def __load_trainer(trainer_path):
        with open(trainer_path, 'rb') as reader:
            return pkl.load(reader)

    @staticmethod
    def __validate_dir_paths(*paths):
        for path in list(paths):
            if not os.path.exists(path):
                raise Exception("Path {} is not found!".format(path))
