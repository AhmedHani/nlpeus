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

import re
import codecs
import utils.text_utils as text_utils
from utils.text_utils import Preprocessor, TextDatasetAnalyzer


class PapersNewsDataProcessing(object):
    
    def __init__(self, news_file, papers_file, pre_processing=False):
        self.news_data = self.__read_news_file(news_file, pre_processing)
        self.papers_data = self.__read_papers_file(papers_file, pre_processing)
        
        self.index2class, self.class2index = {0: 'news', 1: 'papers'}, {'news': 0, 'papers': 1}        

    @staticmethod
    def __read_news_file(news_file, pre_processing):
        news_data = []

        with codecs.open(news_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = line.strip().rstrip()

                if pre_processing:
                    line = Preprocessor.normalize_text(line)
                    line = Preprocessor.replace_apostrophes(line)
                    ## pad and truncate TODO
                
                news_data.append((line, 0))
        
        return news_data

    @staticmethod
    def __read_papers_file(papers_file, pre_processing):
        papers_data = []

        with codecs.open(papers_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = line.strip().rstrip()

                if pre_processing:
                    line = Preprocessor.normalize_text(line)
                    line = Preprocessor.replace_apostrophes(line)
                    ## pad and truncate TODO
                
                papers_data.append((line, 1))

        return papers_data

class SpookyAuthorsDataProcessing(object):
    
    def __init__(self, train_file, test_file, preprocessing=False):
        self.train_data = self.__read_train_file(train_file, preprocessing)
        self.test_data = self.__read_test_file(test_file, preprocessing)

        self.class2index = {'MWS': 0, 'EAP': 1, 'HPL': 2}
        self.index2class = {0: 'MWS', 1: 'EAP', 2: 'HPL'}
    
    @staticmethod
    def __read_train_file(train_file, preprocessing):
        train_data = []
        regex = re.compile("\"(.*?)\"")

        with codecs.open(train_file, 'r', encoding='utf-8') as reader:
            all_lines = reader.readlines()[1:]

            for line in all_lines:
                line_tokens = list(filter(None, regex.findall(line.strip().rstrip())))

                if len(line_tokens) == 3:
                    id_ = line_tokens[0]
                    text = line_tokens[1]
                    author = line_tokens[2]
                else:
                    id_ = line_tokens[0]
                    text = ''.join(line_tokens[1:-1])
                    author = line_tokens[-1]

                if preprocessing:
                    text = Preprocessor.normalize_text(text)
                    text = Preprocessor.replace_apostrophes(text)
                
                train_data.append((id_, text, author))
            
        return train_data

    @staticmethod
    def __read_test_file(test_file, preprocessing):
        test_data = []
        regex = re.compile("\"(.*?)\"")

        with codecs.open(test_file, 'r', encoding='utf-8') as reader:
            all_lines = reader.readlines()[1:]

            for line in all_lines:
                line_tokens = list(filter(None, regex.findall(line.strip().rstrip())))
                
                if len(line_tokens) == 2:
                    id_ = line_tokens[0]
                    text = line_tokens[1]
                else:
                    id_ = line_tokens[0]
                    text = ''.join(line_tokens[1:])

                if preprocessing:
                    text = Preprocessor.normalize_text(text)
                    text = Preprocessor.replace_apostrophes(text)
                
                test_data.append((id_, text))
            
        return test_data
