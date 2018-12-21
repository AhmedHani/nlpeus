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