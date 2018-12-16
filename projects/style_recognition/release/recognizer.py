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
import json
import argparse
import pickle as pkl
from glob import glob
from models.torch_charnn import CharRNN
from common.trainer import SupervisedTrainer
from common.transformations import TextTransformations
from utils.text_utils import TextDatasetAnalyzer, TextEncoder, Preprocessor


class StyleRecognizer(object):

    def __init__(self, production_dir='./projects/style_recognition/shared/production', device='cpu'):
        saved_data = os.path.join(production_dir, 'saved_data')
        saved_model = os.path.join(production_dir, 'saved_model')
        saved_pipeline = os.path.join(production_dir, 'saved_pipeline')

        self.__validate_dir_paths(saved_data, saved_model, saved_pipeline)

        transformations_path = os.path.join(saved_pipeline, 'transformations.pkl')
        self.transformations = self.__load_transformations(transformations_path)

        encoder_path = os.path.join(saved_pipeline, 'encoder.pkl')
        self.encoder = self.__load_encoder(encoder_path)

        model_weights_path = os.path.join(saved_model, 'weights.pt')
        model_path = os.path.join(saved_model, 'model.pkl')
        model_args_path = os.path.join(saved_model, 'args.json')

        self.__validate_dir_paths(model_path, model_weights_path, model_args_path)

        model_kwargs = self.__load_kwargs(model_args_path)
        
        if 'device' in model_kwargs:
            model_kwargs['device'] = device

        self.model = self.__load_model(model_path)(**model_kwargs)
        
        class2index, index2class = self.__load_classes_ids(saved_data)
        
        self.trainer = SupervisedTrainer(self.model, [class2index, index2class])
        self.trainer.load(model_weights_path)

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
    def __load_model(model_path):
        with open(model_path, 'rb') as reader:
            model_class = pkl.load(reader)
        
        return model_class

    @staticmethod
    def __load_kwargs(model_args_path):
        with open(model_args_path, 'r') as reader:
            return dict(json.load(reader))

    @staticmethod
    def __validate_dir_paths(*paths):
        for path in list(paths):
            if not os.path.exists(path):
                raise Exception("Path {} is not found!".format(path))
    
    @staticmethod
    def __load_classes_ids(path):
        class2index, index2class = {}, {}

        for filename in glob(path + '/*.json'):
            if '2index' in filename:
                with open(filename, 'r') as reader:
                    class2index = dict(json.load(reader))
            elif  '2class' in filename:
                with open(filename, 'r') as reader:
                    index2class = dict(json.load(reader))
        
        return class2index, index2class

sr = StyleRecognizer()
