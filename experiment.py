################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model

import pdb


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__model = get_model(config_data, self.__vocab)
        self.__criterion = torch.nn.CrossEntropyLoss()  # cross entropy
        lr = 5e-4  # learning rate
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr)  # adam

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()
        
        
    def decode(self, features, captions):
        x = self.decoder.embed(captions)
        x = torch.cat((features.unsqueeze(1), x), dim=1)
        x, _ = self.model(x)
        x = self.decoder.linear(x)
        

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            
    def get_caption_from_indices(self, indices):
        return ' '.join([self.__vocab.idx2word[idx] for idx in indices])
    
    def reshape(self, tensor, new_shape):
        shape = (tensor.shape[new_shape[0]], tensor.shape[new_shape[1]], tensor.shape[new_shape[2]])
        return torch.reshape(tensor, shape)

    # Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = 0
        # 1164 iterations
        for i, (images, captions, _) in enumerate(self.__train_loader):
            self.__optimizer.zero_grad()
            captions = captions.to('cuda')
            og_captions = captions
            # add a column of zeros to captions
            zeros = torch.zeros(captions.shape[0]).type(torch.LongTensor).to('cuda').unsqueeze(1)
            captions = torch.cat((zeros, captions), axis=1)
            #pdb.set_trace()
            probabilities = self.__model.forward2(images,captions)  # output is tensor of (batch_size, seq_len, vocab_size)
            probabilities = probabilities[:, :-1, :]  # remove the last layer
            
            # TODO code stochastic
            
            # deterministic
            indices = probabilities[0, :, :].argmax(axis=1).cpu().numpy()
            generated_words = self.get_caption_from_indices(indices)
            vocab_size = probabilities.shape[2]
            loss = self.__criterion(probabilities.reshape(-1, vocab_size), og_captions.view(-1))  # Compute the loss here
            loss.backward()
            self.__optimizer.step()
            
            if i%100 == 0:  # print every 10th iteration
                print("Training iteration: %s, loss: %s" % (i, loss.item()))
                original_caption = self.get_caption_from_indices(og_captions[0].cpu().numpy())
                print("Original caption: %s" % original_caption)
                generated_caption = generated_words
                print("Generated caption: %s" % generated_caption)
            training_loss += loss.item()

        return training_loss/len(self.__train_loader)

    # Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0
        with torch.no_grad():
            # 129 iterations
            for i, (images, captions, _) in enumerate(self.__val_loader):
                captions = captions.to('cuda')
                og_captions = captions
               # add a column of zeros to captions
                zeros = torch.zeros(captions.shape[0]).type(torch.LongTensor).to('cuda').unsqueeze(1)
                captions = torch.cat((zeros, captions), axis=1)
                #pdb.set_trace()
                probabilities = self.__model.forward2(images,captions)  # output is tensor of (batch_size, seq_len, vocab_size)
                probabilities = probabilities[:, :-1, :]  # remove the last layer
                vocab_size = probabilities.shape[2]
                loss = self.__criterion(probabilities.reshape(-1, vocab_size), og_captions.view(-1))  # Compute the loss here
                #self.__optimizer.step()
                val_loss += loss.item()
                print("val iteration %s" % i)

        return val_loss/len(self.__val_loader)

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note that you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        import nltk
        temp = 1
        self.__model.eval()
        test_loss = 0
        bleu1_total = 0
        bleu4_total = 0
        iter = 0
        stochastic = False
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                batch_loss = 0
                captions = captions.to('cuda')
                og_captions = captions
                
                # generate captions
                generated_captions = []
                for j, encoded_image in enumerate(images.unsqueeze(1)):
                    generated_caption = []
                    saved_probabilities = []
                    has_ended = False
                    lstm_input = zeros = torch.zeros(1).type(torch.LongTensor).to('cuda')
                    for i in range(captions.shape[1]):
                        if i == 0:
                            lstm_output, (h_i, c_i) = self.__model.LSTMBoth(encoded_image, lstm_input)
                        else:
                            lstm_output, (h_i, c_i) = self.__model.LSTMBoth(encoded_image, lstm_input, (h_i, c_i))
                        lstm_output=lstm_output[:,1,:]
                        lstm_output=lstm_output.unsqueeze(1)
                        probabilities = self.__model.output(lstm_output)
                        vocab_size = probabilities.shape[2]
                        # TODO remove first layer of probabilities
                        saved_probabilities.append(probabilities.reshape(-1, vocab_size))
                        if stochastic:
                            word_index = torch.multinomial(F.softmax(saved_probabilities[-1].squeeze().div(temp)).data, 1).item()
                        else:
                            word_index = probabilities.argmax().item()
                        
                        word = self.__vocab.idx2word[word_index]
                        lstm_input = torch.Tensor([word_index]).type(torch.LongTensor).to('cuda')

                        if word == '<end>':
                            has_ended = True
                        elif word in {'<start>', '<pad>'}:
                            pass
                        else:
                            if not has_ended:
                                generated_caption.append(word)
                    og_caption = og_captions[j].view(-1)
                    batch_loss += self.__criterion(torch.cat(saved_probabilities), og_caption).item()
                    generated_captions.append(generated_caption)
                bleu1_scores = []
                bleu4_scores = []
                for i in range(len(img_ids)):
                    annIds = self.__coco_test.getAnnIds(img_ids[i]);
                    anns = self.__coco_test.loadAnns(annIds)
                    original_captions = []
                    for j in range(len(anns)):
                        original_captions.append(nltk.tokenize.word_tokenize(anns[j]['caption'].lower()))
                        
                    print("Image: %s" % self.__coco_test.loadImgs(img_ids[i])[0]['file_name'])
                    print("Original caption: ")
                    self.__coco_test.showAnns(anns)
                    print("Generated caption: %s" % ' '.join(generated_captions[i]),end='\n\n')
                    bleu1_scores.append(bleu1(original_captions, generated_captions[i]))
                    bleu4_scores.append(bleu4(original_captions, generated_captions[i]))
                avg_bleu1 = np.mean(bleu1_scores)
                avg_bleu4 = np.mean(bleu4_scores)
                test_loss += batch_loss/j
                bleu1_total += avg_bleu1
                bleu4_total += avg_bleu4
                print("Test iteration: %s, batch loss: %s, bleu1: %s, bleu4: %s" % (iter, batch_loss, avg_bleu1, avg_bleu4))
                
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss/iter,
                                                                               bleu1_total/iter,
                                                                               bleu4_total/iter)
        self.__log(result_str)
        return test_loss, bleu1_total, bleu4_total

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
