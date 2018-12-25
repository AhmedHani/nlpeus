import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim


class TorchResBlock2D(nn.Module):

    def __init__(self, in_map, out_map, kernel_size=3, stride=1, padding=1):
        pass
    
    def __new__(cls, in_map, out_map, kernel_size=3, stride=1, padding=1):
        cnn1 = nn.Conv2d(in_channels=in_map, out_channels=out_map, kernel_size=kernel_size, 
                        stride=stride, padding=padding, bias=False)
        bn1 = nn.BatchNorm2d(out_map)
        relu1 = nn.ReLU()

        cnn2 = nn.Conv2d(in_channels=out_map, out_channels=in_map, kernel_size=kernel_size, 
                        stride=stride, padding=padding, bias=False)
        bn2 = nn.BatchNorm2d(in_map)

        block = nn.Sequential(cnn1, bn1, relu1, cnn2, bn2)
        
        return block


class TorchResNet2D(nn.Module):

    def __init__(self, input_size, output_size, n_blocks=4, device='cpu'):
        super().__init__()
        
        if len(input_size) == 2:
            input_size.insert(0, 1)

        self.input_channels, self.height, self.wdith = input_size

        self.net = self.build(n_blocks)

        self.loss = nn.NLLLoss()
        self.optimzer = optim.Adam(self.parameters(), lr=1e-3)

        self.device = device
        torch.device(device)

        self.to(device)
    
    def build(self, n_blocks, base_expansion=64):
        blocks = []

        cnn1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=base_expansion,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()) 

        blocks.append(cnn1)

        self.output_shape = self.__get_output_shape(input_dim=[self.input_channels, self.height, self.wdith],
                                                    ochannels=base_expansion, kernel_size=3, stride=1, padding=1, flatten=False)

        for i in range(0, n_blocks):
            blocks.append(TorchResBlock2D(
                in_map=base_expansion,
                out_map=base_expansion * 2,
                kernel_size=3,
                stride=1
            ))

            self.output_shape = self.__get_output_shape(input_dim=self.output_shape, 
                                                        ochannels=base_expansion * 2, kernel_size=3,
                                                        stride=1, padding=1, flatten=False)
            base_expansion *= 2
        
        avgpool = nn.AvgPool2d(kernel_size=2)
        self.output_shape = self.__get_output_shape(input_dim=self.output_shape, ochannels=base_expansion / 2, kernel_size=2,
                                                       stride=None, padding=0, flatten=True)

        blocks.append(avgpool)
        self.output = nn.Linear(self.output_shape, self.output_size)
        
        blocks.append(self.output)

        return blocks
        
    def forward(self, input):
        input = torch.from_numpy(input).to(self.device).float()
        input = input.view(input.size(0), self.input_channels, input.size(2), input.size(1))

        current = self.net[0](input)
        prev = current

        for i in range(1, len(self.net) - 2):
            current = self.net[i](current)
            current += prev
            current = nn.ReLU()(current)

            prev = current
        
        current = self.net[-2](current)
        current = self.net[-1](current)

        output = F.log_softmax(current, dim=1)

        return output
    
    def predict_classes(self, input):
        output = self.forward(input)[0]

        return output.cpu().max(1)[1].data.numpy()

    def predict_probs(self, input):
        input = torch.LongTensor(input).to(self.device)
        batch_size = input.size(0)

        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded, self.init_hidden(batch_size))

        output = self.decoder(hidden[0].view(hidden[0].size(1), hidden[0].size(2)))

        output = F.softmax(output, dim=1)

        return output.cpu().max(1)[0].data.numpy()

    def loss_function(self, predicted, target):
        target = torch.LongTensor(target).to(self.device)

        return self.loss(predicted, target)

    def optimize(self):
        return self.optimzer.step()

    def calculate_gradient(self, prediction, target):
        self.zero_grad()
        loss = self.loss_function(prediction, target)

        loss_value = loss.item()
        loss.backward()

        return float(loss_value)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

        return True

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

        return True
    
    def args(self):
        return self.kwargs

    @staticmethod
    def __get_output_shape(input_dim, ochannels, kernel_size, stride=1, padding=1, flatten=False):
        """
            Input Shape: nChannels x Width x Height
            Filter = F
            Stride = S
            Padding = P
            Output Shape: Width = ((Width - F + (2*P))/ S) + 1, Height = ((Height - F + (2*P))/ S) + 1
        """

        stride = kernel_size if stride is None else stride
        nchannels, height, width = input_dim[0], input_dim[1], input_dim[2]

        new_width = int(((width - kernel_size + (2 * padding)) / stride) + 1)
        new_height = int(((height - kernel_size + (2 * padding)) / stride) + 1)

        return [ochannels, new_height, new_width] if not flatten else ochannels * int(new_height) * int(new_width)