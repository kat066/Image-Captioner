################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
import torch.nn as nn
from torch.nn import Embedding, LSTM
import torchvision.models as models
import torch

import pdb

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    vocab_size = len(vocab)

    return Model(hidden_size,embedding_size,model_type,vocab_size)



class Model(nn.Module):
    def __init__(self,hidden_size,embedding_size,model_type,vocab_size):
        super(Model, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad_(False)
        self.resnet.fc = torch.nn.Linear(2048, embedding_size)
        self.resnet = self.resnet.to('cuda')
        
        self.embed = torch.nn.Embedding(vocab_size, embedding_size)
        self.embed = self.embed.to('cuda')
        self.lstm = LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)
        
        self.output = torch.nn.Linear(hidden_size, vocab_size)
        self.output = self.output.to('cuda')
     
    def forward(self, images, captions):
        images = images.to('cuda')
        im = self.resnet(images)
        im = im.unsqueeze(1)
        ca = self.embed(captions)
        combined = torch.cat((im,ca), 1)
        output,(h_i,c_i) = self.lstm(combined)
        out = self.output(output)
        return out
    
    def forward2(self, images, captions):
        images = images.to('cuda')
        im = self.resnet(images)
        im = im.unsqueeze(1)
        ca = self.embed(captions)
        # images shape: (64, 1, 300)
        # captions shape: (64, 20, 300)
        combined = torch.cat((im, ca[:, 0, :].unsqueeze(1)), 1)
        for i in range(1, ca.shape[1]):
            temp = torch.cat((im, ca[:, i, :].unsqueeze(1)), 1)
            combined = torch.cat((combined, temp), 1)
        output, (h_i,c_i) = self.lstm(combined)
        out = self.output(output)
        combined = out[:, 1, :].unsqueeze(1)
        for i in range(3, out.shape[1], 2):
            combined = torch.cat((combined, out[:, i, :].unsqueeze(1)), 1)
        return combined
    
    def LSTMImg(self, images):
        im = images.to('cuda')
        im = self.resnet(im)
        im = im.unsqueeze(1)
        return self.lstm(im)
    
    def LSTMCap(self, captions, hidden_state):
        ca = self.embed(captions)
#         pdb.set_trace()
        return self.lstm(ca, hidden_state)

    def LSTMBoth(self, images, captions, hidden_state=None):
        im = images.to('cuda')
        im = self.resnet(im)
        im = im.unsqueeze(1)
        
        ca = self.embed(captions)
        ca = ca.unsqueeze(0)
        
        combined = torch.cat((im, ca), dim=1)
        if hidden_state:
            return self.lstm(combined, hidden_state)
        else:
            return self.lstm(combined)


        