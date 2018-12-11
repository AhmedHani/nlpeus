import re
import copy as cp
from functools import reduce
import os
import codecs
import sys
from collections import Counter


class Preprocessor:

    @staticmethod
    def remove_extra_spaces(text):
        if isinstance(text, list):
            return [re.sub(' +', ' ', one) for one in text]

        return re.sub(' +', ' ', text)
    
    @staticmethod
    def normalize_text(text):
        if isinstance(text, list):
            return [one.lower() for one in text]

        return ' '.join([word.lower() for word in text.split()])
    
    @staticmethod
    def replace_apostrophes(text):
        apostrophes_mapping = {
            '\'s': ' is',
            '\'ve': ' have',
            'n\'t': ' not',
            '\'d': ' would',
            '\'m': ' am',
            '\'ll': ' will',
            '\'re': ' are'
        }

        # elegant! https://stackoverflow.com/a/9479972
        if isinstance(text, list):
            return [reduce(lambda a, kv: a.replace(*kv), apostrophes_mapping.items(), s) for s in text]
    
        return reduce(lambda a, kv: a.replace(*kv), apostrophes_mapping.items(), text)

    @staticmethod
    def remove_punctuations(text):
        if isinstance(text, list):
            return [re.sub(r'[^\w\s]', '', one) for one in text]

        return re.sub(r'[^\w\s]', '', text) 
    
    @staticmethod
    def remove_custom_chars(text, chars_list):
        if isinstance(text, list):
            return [re.sub("|".join(chars_list), "", one) for one in text]

        return re.sub("|".join(chars_list), "", text)
    
    @staticmethod
    def remove_stop_words(text):
        nltk_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                            "you'd", 'your',
                            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
                            'herself', 'it',
                            "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
                            'whom', 'this',
                            'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                            'have', 'has', 'had',
                            'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                            'as',
                            'until',
                            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                            'during',
                            'before',
                            'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                            'again',
                            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                            'each', 'few',
                            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                            'too', 'very',
                            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
                            'o',
                            're', 've',
                            'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
                            "hadn't", 'hasn',
                            "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                            'needn',
                            "needn't",
                            'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                            'wouldn',
                            "wouldn't", 'I']

        if isinstance(text, list):
            return [' '.join([word for word in one.split() if word not in nltk_stop_words]) for one in text]

        return ' '.join([word for word in text.split(' ') if word not in nltk_stop_words])
    
    @staticmethod
    def clean_words(text):

        def _clean(one_text):
            clean_string = []

            one_text_words = one_text.split()

            for word in one_text_words:
                clean_word = ""

                for char in word:
                    if 65 <= ord(char) <= 122 or char.isdigit():
                        clean_word += char

                clean_string.append(clean_word)

            clean_string = ' '.join(clean_string)

            return clean_string

        if isinstance(text, list):
            return [_clean(one) for one in text]

        return _clean(text)
    
    @staticmethod
    def word_based_pad(sentences_list, size=None, token='PAD'):
        sentences = cp.deepcopy(sentences_list)

        if size is None:
            size = max([len(sentence.split()) for sentence in sentences])

        for i, sentence in enumerate(sentences):
            sentence_tokens = sentence.split()

            while len(sentence_tokens) < size:
                sentence_tokens.append(token)

            sentences[i] = ' '.join(sentence_tokens)

        return sentences

    @staticmethod
    def word_based_truncate(sentences_list, size):
        sentences = cp.deepcopy(sentences_list)

        for i, sentence in enumerate(sentences):
            sentence_tokens = sentence.split()
            sentences[i] = ' '.join(sentence_tokens[0:size])

        return sentences
    
    @staticmethod
    def char_based_pad(sentences_list, size=None, token='#'):
        sentences = cp.deepcopy(sentences_list)

        if size is None:
            size = max([len(list(sentence)) for sentence in sentences])

        for i, sentence in enumerate(sentences):
            sentence_tokens = list(sentence)

            while len(sentence_tokens) < size:
                sentence_tokens.append(token)

            sentences[i] = ''.join(sentence_tokens)

        return sentences
    
    @staticmethod
    def chat_based_truncate(sentences_list, size):
        sentences = cp.deepcopy(sentences_list)

        for i, sentence in enumerate(sentences):
            sentence_tokens = list(sentence)
            sentences[i] = ' '.join(sentence_tokens[0:size])

        return sentences


class TextAnalyzer:

    def __init__(self, data, outpath='stdout'):
        self.data = data

        self.out = self.__set_output_location(outpath)
    
    def all(self, n_instances=True,
                  avg_n_words=True, 
                  avg_n_chars=True,
                  n_unique_words=True,
                  n_unique_chars=True,
                  words_freqs=True,
                  chars_freqs=True):
        average_words = 0
        words, chars = {}, {}

        for sentence in self.data:
            sentence_tokens = sentence.split()
            chars_tokens = list(sentence_tokens)
            
            average_words += len(sentence_tokens)
            average_chars += len(chars_tokens)

            for word in sentence_tokens:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1

            for char in chars_tokens:
                if char in chars:
                    chars[char] += 1
                else:
                    chars[char] = 1

        if n_instances:
            self.out.write('number of instances: {}\n'.format(len(self.data)))
        
        if avg_n_words:
            self.out.write('average number of words: {}\n'.format(average_words // len(self.data)))
        
        if avg_n_chars:
            self.out.write('average number of chars: {}\n'.format(average_chars // len(self.data)))
        
        if n_unique_words:
            self.out.write('number of unique words: {}\n'.format(len(words)))
        
        if n_unique_chars:
            self.out.write('number of unique chars: {}\n'.format(len(chars)))
        
        if words_freqs:
            words = Counter(words).most_common(len(words))

            freqs_format = '\n'.join(['\t\t' + key + ': ' + str(value) for key, value in words])
            self.out.write('words frequencies: \n{}\n\n'.format(freqs_format))
        
        if chars_freqs:
            chars = Counter(chars).most_common(len(chars))

            freqs_format = '\n'.join(['\t\t' + key + ': ' + str(value) for key, value in chars])
            self.out.write('chars frequencies: \n{}\n\n'.format(freqs_format))

    def n_instances(self):
        self.out.write('number of instances: {}\n'.format(len(self.data)))
    
    def avg_n_words(self):
        average_words = sum([len(sentence.split()) for sentence in self.data]) // len(self.data)

        self.out.write('average number of words: {}\n'.format(average_words)) 
    
    def avg_n_chars(self):
        average_chars = sum([len(list(sentence)) for sentence in self.data]) // len(self.data)

        self.out.write('average number of words: {}\n'.format(average_chars))

    def n_unique_words(self):
        words = set()

        for sentence in self.data:
            for word in sentence.split():
                words.add(word)
        
        self.out.write('number of unique words: {}\n'.format(len(words)))
    
    def n_unique_chars(self):
        chars = set()

        for sentence in self.data:
            for char in list(sentence):
                chars.add(char)
        
        self.out.write('number of unique chars: {}\n'.format(len(chars)))

    def words_freqs(self):
        words = {}

        for sentence in self.data:
            for word in sentence.split():
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
        
        words = Counter(words).most_common(len(words))

        freqs_format = '\n'.join(['\t\t' + key + ': ' + str(value) for key, value in words])
        self.out.write('words frequencies: \n{}\n\n'.format(freqs_format))
    
    def chars_freqs(self):
        chars = {}

        for sentence in self.data:
            for char in list(sentence):
                if char in chars:
                    chars[char] += 1
                else:
                    chars[char] = 1
        
        chars = Counter(chars).most_common(len(chars))

        freqs_format = '\n'.join(['\t\t' + key + ': ' + str(value) for key, value in chars])
        self.out.write('chars frequencies: \n{}\n\n'.format(freqs_format))

    @staticmethod
    def __set_output_location(outpath):
        if outpath == 'stdout':
            return sys.stdout
        
        if os.path.exists(outpath):
            raise Exception('file {} not existed!'.format(outpath))
        
        return codecs.open(outpath, 'w', encoding='utf-8')


class TextDatasetAnalyzer:

    def __init__(self, data, data_axis, index2class, outpath='stdout'):
        if isinstance(data_axis['text'], int):
            self.data = [item[data_axis['text']] for item in data]
        else:
            self.data = []

            for index in data_axis['text']:
                self.data += [item[index] for item in data]
            
        self.labels = [item[data_axis['label']] for item in data]
        self.index2class = index2class

        self.out = self.__set_output_location(outpath)
    
    def all(self, n_instances=True,
                  avg_n_words=True, 
                  avg_n_chars=True,
                  n_unique_words=True,
                  n_unique_chars=True,
                  words_freqs=True,
                  chars_freqs=True,
                  n_samples_per_class=True):
        average_words, average_chars = 0, 0
        words, chars = {}, {}

        for sentence in self.data:
            sentence_tokens = sentence.split()
            chars_tokens = list(sentence)
            
            average_words += len(sentence_tokens)
            average_chars += len(chars_tokens)

            for word in sentence_tokens:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1

            for char in chars_tokens:
                if char in chars:
                    chars[char] += 1
                else:
                    chars[char] = 1

        if n_instances:
            self.out.write('number of instances: {}\n'.format(len(self.data)))
        
        if avg_n_words:
            self.out.write('average number of words: {}\n'.format(average_words // len(self.data)))
        
        if avg_n_chars:
            self.out.write('average number of chars: {}\n'.format(average_chars // len(self.data)))
        
        if n_unique_words:
            self.out.write('number of unique words: {}\n'.format(len(words)))
        
        if n_unique_chars:
            self.out.write('number of unique chars: {}\n'.format(len(chars)))
        
        if n_samples_per_class:
            classes = Counter(self.labels).most_common(len(self.labels))

            freqs_format = '\n'.join(['\t\t' + self.index2class[key] + ': ' + str(value) for key, value in classes])
            self.out.write('classes frequencies: \n{}\n\n'.format(freqs_format))
        
        if words_freqs:
            words = Counter(words).most_common(len(words))

            freqs_format = '\n'.join(['\t\t' + key + ': ' + str(value) for key, value in words])
            self.out.write('words frequencies: \n{}\n\n'.format(freqs_format))
        
        if chars_freqs:
            chars = Counter(chars).most_common(len(chars))

            freqs_format = '\n'.join(['\t\t' + key + ': ' + str(value) for key, value in chars])
            self.out.write('chars frequencies: \n{}\n\n'.format(freqs_format))

    def n_instances(self):
        self.out.write('number of instances: {}\n'.format(len(self.data)))
    
    def avg_n_words(self):
        average_words = sum([len(sentence.split()) for sentence in self.data]) // len(self.data)

        self.out.write('average number of words: {}\n'.format(average_words)) 
    
    def avg_n_chars(self):
        average_chars = sum([len(list(sentence)) for sentence in self.data]) // len(self.data)

        self.out.write('average number of words: {}\n'.format(average_chars))

    def n_unique_words(self):
        words = set()

        for sentence in self.data:
            for word in sentence.split():
                words.add(word)
        
        self.out.write('number of unique words: {}\n'.format(len(words)))
    
    def n_unique_chars(self):
        chars = set()

        for sentence in self.data:
            for char in list(sentence):
                chars.add(char)
        
        self.out.write('number of unique chars: {}\n'.format(len(chars)))

    def words_freqs(self):
        words = {}

        for sentence in self.data:
            for word in sentence.split():
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
        
        words = Counter(words).most_common(len(words))

        freqs_format = '\n'.join(['\t\t' + key + ': ' + str(value) for key, value in words])
        self.out.write('words frequencies: \n{}\n\n'.format(freqs_format))
    
    def chars_freqs(self):
        chars = {}

        for sentence in self.data:
            for char in list(sentence):
                if char in chars:
                    chars[char] += 1
                else:
                    chars[char] = 1
        
        chars = Counter(chars).most_common(len(chars))

        freqs_format = '\n'.join(['\t\t' + key + ': ' + str(value) for key, value in chars])
        self.out.write('chars frequencies: \n{}\n\n'.format(freqs_format))

    def n_samples_per_class(self):
        classes = Counter(self.labels).most_common(len(self.labels))

        freqs_format = '\n'.join(['\t\t' + self.index2class[key] + ': ' + str(value) for key, value in classes])
        self.out.write('classes frequencies: \n{}\n\n'.format(freqs_format))

    @staticmethod
    def __set_output_location(outpath):
        if outpath == 'stdout':
            return sys.stdout
        
        if not os.path.exists(os.path.dirname(outpath)):
            raise Exception('file {} not existed!'.format(outpath))
        
        return codecs.open(outpath, 'w', encoding='utf-8')


class TextEncoder:

    @staticmethod
    def word_based_embedding(text, embedding_model, flatten=False):
        if isinstance(text, list):
            sentences_vectors = []

            for sentence in text:
                sentence_tokens = sentence.split()

                sentence_matrix = []

                for word in sentence_tokens:
                    if word in embedding_model:
                        vec = embedding_model[word]
                    else:
                        vec = embedding_model['unk']

                    sentence_matrix.append(vec)

                if flatten:
                    sentence_matrix = sum(sentence_matrix, [])

                sentences_vectors.append(sentence_matrix)

            return sentences_vectors
        else:
            sentences_vectors = []
            sentence_tokens = text.split()

            sentence_matrix = []

            for word in sentence_tokens:
                if word in embedding_model:
                    vec = embedding_model[word]
                else:
                    vec = embedding_model['unk']

                sentence_matrix.append(vec)

            if flatten:
                sentence_matrix = sum(sentence_matrix, [])

            return sentence_matrix
    
    @staticmethod
    def char_based_embedding(text, embedding_model, flatten=False):
        if isinstance(text, list):
            text_matrix = []

            for sentence in text:
                chars_tokens = list(sentence)
                sentence_matrix = []
            
                for char in chars_tokens:
                    if char in embedding_model:
                        vec = embedding_model[char]
                    else:
                        vec = embedding_model['*']
                
                    sentence_matrix.append(vec)
            
                if flatten:
                    sentence_matrix = sum(sentence_matrix, [])
    
                text_matrix.append(sentence_matrix)

            return text_matrix
        else:
            sentence_matrix = []

            for char in list(text):
                if char in embedding_model:
                    vec = embedding_model[char]
                else:
                    vec = embedding_model['*']
                
                sentence_matrix.append(vec)
        
            if flatten:
                sentence_matrix = sum(sentence_matrix, [])
        
            return sentence_matrix

    @staticmethod
    def sentence_based_embedding(text, inference_func):
        if isinstance(text, list):
            return [inference_func(sentence) for sentence in text]
        
        return inference_func(text)
