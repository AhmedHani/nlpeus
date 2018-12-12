import argparse

from models.torch_charnn import CharRNN
from style_recognition.research.data_processing import DataProcessing
from style_recognition.research.trainer import Trainer
from utils.data_utils import Batcher
from utils.experiment_utils import Experiment
from utils.text_utils import TextDatasetAnalyzer
from utils.text_utils import TextEncoder, Preprocessor
import functools
from style_recognition.research.local_driver import __file__ as local_driver_file


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

#dataset_analyzer = TextDatasetAnalyzer(data=data, data_axis={'text': 0, 'label': 1},
#                                       index2class=index2class,
#                                       outpath='./style_recognition/datasets/all_data_analysis.log')

#dataset_analyzer.all()
#del dataset_analyzer

batcher = Batcher(data=data, batch_size=batch_size, with_shuffle=True, divide_train_valid_test=True)
dataset_analyzer = TextDatasetAnalyzer(data=batcher.train_data, data_axis={'text': 0, 'label': 1},
                                       index2class=index2class, outpath='stdout')

char2index, index2char = dataset_analyzer.get_chars_ids()
chars_freqs = dataset_analyzer.get_chars_freqs()

model = CharRNN(input_size=len(char2index), output_size=len(class2index) - 1)

experiment = Experiment(
    total_samples=len(data),
    total_training_samples=batcher.total_train_samples,
    total_valid_samples=batcher.total_valid_samples,
    total_test_samples=batcher.total_test_samples,
    model_name=model.__class__.__name__,
    epochs=epochs,
    batch_size=batch_size,
    number_classes=len(class2index),
    input_length=max_charslen,
    device=device,
    author_name='A.H. Al-Ghidani'
)

experiment_name = experiment.create(local_driver_file)
saved_model_path = experiment.saved_model_dir
saved_data_path = experiment.saved_data_dir
eval_file_path = experiment.eval_file_path
pickle_file_path = experiment.pickle_file_path
learning_curve_image = experiment.learning_curve_image

trainer = Trainer(model, classes=[class2index, index2class])
text_encoder = TextEncoder(char2indexes=char2index, modelname='char_one_hot')
transformations = [functools.partial(Preprocessor.char_based_pad, size=max_charslen),
                   functools.partial(Preprocessor.chat_based_truncate, size=max_charslen)]

experiment.run(trainer, batcher, encoder=text_encoder, transformations=transformations, data_axis={'X': 0, 'Y': 1})
