import argparse

from style_recognition.data_processing import DataProcessing
from utils.data_utils import Batcher
from utils.experiment_utils import Experiment
from models.torch_charnn import CharRNN
from utils.text_utils import Preprocessor, TextDatasetAnalyzer, TextAnalyzer


parser = argparse.ArgumentParser(description='Style Recognition training playground')

parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--epochs', type=int, default=15, help='number of training epochs')
parser.add_argument('--max-charslen', type=int, default=70, help='max chars length that will be fed to the network')
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

dp = DataProcessing(news_file='./style_recognition/datasets/news.txt',
                    papers_file='./style_recognition/datasets/paper.txt',
                    pre_processing=False)

data = dp.news_data + dp.papers_data
class2index, index2class = dp.class2index, dp.index2class

dataset_analyzer = TextDatasetAnalyzer(data=data, data_axis={'text': 0, 'label': 1},
                                       index2class=index2class,
                                       outpath='./style_recognition/outs/all_data_analysis.log')

dataset_analyzer.all()
del dataset_analyzer

batcher = Batcher(data=data, batch_size=batch_size, with_shuffle=True, divide_train_valid_test=True)
dataset_analyzer = TextDatasetAnalyzer(data=batcher.train_data, 
                                       data_axis={'text': 0, 'label': 1},
                                       index2class=index2class,
                                       outpath='./style_recognition/outs/train_data_analysis.log')

char2index, index2char = dataset_analyzer.get_chars_ids()
chars_freqs = dataset_analyzer.get_chars_freqs()

model = CharRNN(input_size=len(char2index), output_size=len(class2index) - 1)

experiment = Experiment(
    total_samples=len(data),
    total_training_samples=batcher.total_train_samples,
    total_valid_samples=batcher.total_valid_samples,
    total_test_samples=batcher.total_test_samples,
    model=model,
    batcher=batcher,
    model_name=model.__class__.__name__,
    epochs=epochs,
    batch_size=batch_size,
    number_classes=len(class2index),
    input_length=max_charslen,
    device=device,
    author_name='A.H. Al-Ghidani'
)
