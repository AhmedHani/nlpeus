import re
import copy as cp
from functools import reduce
import os
import codecs
import sys
from collections import Counter
from models.torch_fb_infersent import InferSent


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
            sentences[i] = ''.join(sentence_tokens[0:size])

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
    
    def get_words_ids(self):
        word2index, index2word = {}, {}
        index = 0 

        for sentence in self.data:
            for word in sentence.split():
                if word not in word2index:
                    word2index[word] = index
                    index2word[index] = word

                    index += 1
        
        return word2index, index2word
    
    def get_words_freqs(self):
        words = {}

        for sentence in self.data:
            for word in sentence.split():
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
        
        return words
    
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
    
    def get_chars_ids(self):
        char2index, index2char = {}, {}
        index = 0 

        for sentence in self.data:
            for char in list(sentence):
                if char not in char2index:
                    char2index[char] = index
                    index2char[index] = char

                    index += 1
        
        return char2index, index2char
    
    def get_chars_freqs(self):
        chars = {}

        for sentence in self.data:
            for char in list(sentence):
                if char in chars:
                    chars[char] += 1
                else:
                    chars[char] = 1
        
        return chars

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


class TextEncoder(object):

    def __init__(self, vocab2indexes=None, char2indexes=None, modelname='char_one_hot'):
        if isinstance(modelname, str):
            if modelname == 'char_one_hot':
                self.model = _CharOneHotLoader(char2indexes)
            elif modelname == 'word_one_hot':
                self.model = _WordOneHotLoader(vocab2indexes)
            elif modelname == 'char_embedding':
                self.model = _CharEmbeddingLoader(modelname)
            else:
                self.model = _WordEmbeddingLoader(modelname)
        else:
            import inspect

            if isinstance(modelname, object) or inspect.isclass(modelname):
                self.model = modelname

    def encode(self, text):
        return self.model.encode(text)


class _WordEmbeddingLoader(object):

    def __init__(self, model_name):
        self.__models = ('word2vec', 'fasttext', 'glove')

        assert model_name in self.__models

        self.__load_embeddings(model_name)

    def encode(self, text):
        if isinstance(text, list):
            sentences_vectors = []

            for sentence in text:
                sentence_tokens = sentence.split()

                sentence_matrix = []

                for word in sentence_tokens:
                    vec = self[word]

                    sentence_matrix.append(vec)

                sentences_vectors.append(sentence_matrix)

            return sentences_vectors
        else:
            sentence_tokens = text.split()

            sentence_matrix = []

            for word in sentence_tokens:
                vec = self[word]

                sentence_matrix.append(vec)

            return sentence_matrix

    def __load_embeddings(self, model_name):
        # floydhub input path support

        if model_name == 'word2vec':
            try:
                self.__model, self.__vocab_size, self.__embedding_size = self.__load_word2vec_model(
                    './data_utils/GoogleNews-vectors-negative300.bin')
            except:
                self.__model, self.__vocab_size, self.__embedding_size = self.__load_word2vec_model(
                    '../input/word2vecgooglenewsvectors/GoogleNews-vectors-negative300.bin')
        elif model_name == 'fasttext':
            try:
                self.__model, self.__vocab_size, self.__embedding_size = self.__load_fasttext_model(
                    './data_utils/crawl-300d-2M.vec')
            except:
                self.__model, self.__vocab_size, self.__embedding_size = self.__load_fasttext_model(
                    '../input/fasttextcrawl300d2m/crawl-300d-2M.vec')
        elif model_name == 'glove':
            try:
                self.__model, self.__vocab_size, self.__embedding_size = self.__load_fasttext_model(
                    './data_utils/glove.6B.300d.txt')
            except:
                self.__model, self.__vocab_size, self.__embedding_size = self.__load_fasttext_model(
                    '../input/glove6b/glove.6B.300d.txt')

        print(model_name, "Loaded!")
        print('Vocab Size', self.__vocab_size)
        print('Embedding Dim', self.__embedding_size)

    @staticmethod
    def __load_word2vec_model(fname):
        import numpy as np

        word_vecs = {}

        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size

            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1).decode('latin-1')
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)

                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')

        return word_vecs, vocab_size, layer1_size

    @staticmethod
    def __load_fasttext_model(fname):
        word_vecs = {}

        with open(fname, 'r') as reader:
            for line in reader:
                if vocab_size == -1:
                    vocab_size, embedding_size = line.strip().rstrip().split()
                    continue

                line = line.strip().rstrip()

                if line == "":
                    continue

                line_tokens = line.split(" ")
                word = str(line_tokens[0])
                vec = list(map(lambda v: float(v), line_tokens[1:]))

                if word not in word_vecs:
                    word_vecs[word] = vec

                embedding_size = len(vec)

        return word_vecs, vocab_size, embedding_size

    def __getitem__(self, item):
        if item in self.__model:
            return self.__model[item]
        else:
            vec = [0.0] * self.__embedding_size

            return vec

    def __len__(self):
        return len(self.__model)


class _WordOneHotLoader(object):

    def __init__(self, word2indexes):
        self.word2indexes = word2indexes
        self.word_size = len(word2indexes)
        self.temp_vec = self.__one_hot_vec(self.word_size)

    def encode(self, text):
        if isinstance(text, list):
            text_matrix = []

            for sentence in text:
                sentence_matrix = []

                for word in sentence.split():
                    vec = self[word]

                    sentence_matrix.append(vec)

                text_matrix.append(sentence_matrix)

            return text_matrix
        else:
            sentence_matrix = []

            for word in text.split():
                vec = self[word]

                sentence_matrix.append(vec)

            return sentence_matrix

    @staticmethod
    def __one_hot_vec(word_size):
        return [0.0] * word_size

    def __getitem__(self, item):
        import copy as cp

        if item in self.word2indexes:
            vec = cp.deepcopy(self.temp_vec)
            vec[self.word2indexes[item]] = 1.0

            return vec
        else:
            return self.temp_vec

    def __len__(self):
        return self.word_size


class _CharEmbeddingLoader(object):

    def __init__(self, *args):
        self.args = args


class _CharOneHotLoader(object):

    def __init__(self, char2indexes):
        self.char2indexes = char2indexes
        self.char_size = len(char2indexes)
        self.temp_vec = self.__one_hot_vec(self.char_size)

    def encode(self, text):
        if isinstance(text, list):
            text_matrix = []

            for sentence in text:
                chars_tokens = list(sentence)
                sentence_matrix = []

                for char in chars_tokens:
                    vec = self[char]

                    sentence_matrix.append(vec)

                text_matrix.append(sentence_matrix)

            return text_matrix
        else:
            sentence_matrix = []

            for char in list(text):
                vec = self[char]

                sentence_matrix.append(vec)

            return sentence_matrix

    @staticmethod
    def __one_hot_vec(char_size):
        return [0.0] * char_size

    def __getitem__(self, item):
        import copy as cp

        if item in self.char2indexes:
            vec = cp.deepcopy(self.temp_vec)
            vec[self.char2indexes[item]] = 1.0

            return vec
        else:
            return self.temp_vec

    def __len__(self):
        return self.char_size


