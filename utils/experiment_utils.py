import os
import uuid
import shutil
import codecs
import datetime


class Experiment(object):

    def __init__(self, total_samples,
                 total_training_samples,
                 total_valid_samples,
                 total_test_samples,
                 model,
                 batcher,
                 model_name,
                 epochs,
                 batch_size,
                 number_classes,
                 input_length,
                 device,
                 author_name=None):
        self.total_samples = total_samples
        self.total_training_samples = total_training_samples
        self.total_valid_samples = total_valid_samples
        self.total_test_samples = total_test_samples
        self.model = model
        self.batcher = batcher
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.number_classes = number_classes
        self.input_length = input_length
        self.device = device
        self.author_name = author_name

        self.saved_model_dir = None
        self.saved_data_dir = None
        self.eval_file_path = None
        self.pickle_file_path = None
        self.info_file_path = None
        self.learning_curve_image = None

    def create(self, research_interface):
        project_dir = os.path.dirname(os.path.dirname(research_interface))
        experiment_resources = os.path.join(project_dir, 'shared')
        experiment_resources = os.path.join(experiment_resources, 'experiments')

        experiment_name = 'nclasses({})ninput({})model({})epochs({})batchsize({})device({})'.format(
            self.number_classes,
            self.input_length,
            self.model_name,
            self.epochs,
            self.batch_size,
            self.device
        )

        experiment_dir = os.path.join(experiment_resources, experiment_name)

        if os.path.exists(experiment_dir):
            print('this experiment setup is already done before, do you want to repeat it? [yes/no]')
            answer = str(input())

            if answer.strip().rstrip().lower() == 'yes' or answer.strip().rstrip().lower() == 'y':
                print('do you want to overwrite the previous experiment? [yes/no]')
                answer = str(input())

                if answer.strip().rstrip().lower() == 'yes' or answer.strip().rstrip().lower() == 'y':
                    shutil.rmtree(experiment_dir)
                else:
                    print('write a suffix for the new experiment name')
                    answer = str(input())
                    experiment_dir = os.path.join(experiment_resources, experiment_name + '_{}'.format(answer))
            else:
                print('experiment will be terminated')
                exit()

        os.mkdir(experiment_dir)
        print('experiment location: {}\n'.format(experiment_dir))

        self.saved_model_dir = os.path.join(experiment_dir, 'saved_model')
        os.mkdir(self.saved_model_dir)

        self.saved_data_dir = os.path.join(experiment_dir, 'saved_data')
        os.mkdir(self.saved_data_dir)

        self.eval_file_path = os.path.join(experiment_dir, 'eval.log')
        self.pickle_file_path = os.path.join(experiment_dir, 'eval.pkl')
        self.info_file_path = os.path.join(experiment_dir, 'info.txt')
        self.learning_curve_image = os.path.join(experiment_dir, 'learning_curve.png')

        with codecs.open(self.info_file_path, 'w', encoding='utf-8') as writer:
            writer.write('author: {}\n'.format(self.author_name))

            project_name = os.path.basename(project_dir)
            writer.write('project: {}\n'.format(project_name))

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            writer.write('date and time: {}\n\n'.format(current_time))

            writer.write('experiment setup:\n')
            writer.write('\t total training samples: {}\n'.format(self.total_training_samples))
            writer.write('\t total validation samples: {}\n'.format(self.total_valid_samples))
            writer.write('\t total testing samples: {}\n'.format(self.total_test_samples))
            writer.write('\t model: {}\n'.format(self.model_name))
            writer.write('\t epochs: {}\n'.format(self.epochs))
            writer.write('\t batch size: {}\n'.format(self.batch_size))
            writer.write('\t number of classes: {}\n'.format(self.number_classes))
            writer.write('\t input length: {}\n'.format(self.input_length))
            writer.write('\t device: {}\n'.format(self.device))

        return experiment_dir

    #def do(self):
