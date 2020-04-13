import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

USE_CUDA = True
from IPython.display import clear_output
import matplotlib.pyplot as plt


class Gen_data(Dataset):
    
    def __init__(self, n_node, n_sam_size, seed=1):
        super(Gen_data, self).__init__()
        self.data = []
        torch.manual_seed(seed)
        for i in range(n_sam_size):
            dat = torch.FloatTensor(2, n_node).uniform_(0, 1)
            self.data.append(dat)
        
        self.size = len(self.data)
    
    def __getitem__(self, index):
        da_ind = self.data[index]
        return da_ind
    
    def __len__(self):
        return len(self.data)


train_size = 1280000
v_size = 5000
data_20 = Gen_data(20, train_size)
val_20 = Gen_data(20, v_size)


# data_50 = Gen_data(50, train_size)


class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau', use_cuda=USE_CUDA):
        super(Attention, self).__init__()
        
        self.use_tanh = use_tanh
        self.C = C
        self.name = name
        
        if name == 'Bahdanau':
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1, 1)
            
            V = torch.FloatTensor(hidden_size)
            if use_cuda:
                V = V.cuda()
            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

    def forward(self, query, ref):
        """
            Args:
            query: [batch_size x hidden_size]
            ref:   ]batch_size x seq_len x hidden_size]
        """
        batch_size = ref.size(0)
        seq_len = ref.size(1)

        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref = self.W_ref(ref)  # [batch_size x hidden_size x seq_len]
            expanded_query = query.repeat(1, 1, seq_len)  # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, F.tanh(expanded_query + ref)).squeeze(1)

        elif self.name == 'Dot':
            query = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2)  # [batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)

        else:
            raise NotImplementedError

        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits
        return ref, logits

# transform nodes  to vector
class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, use_cuda=USE_CUDA):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda

        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size))
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)
        embedded = []
        inputs = inputs.unsqueeze(1)
        for i in range(seq_len):
            embedded.append(torch.bmm(inputs[:, :, :, i].float(), embedding))
        embedded = torch.cat(embedded, 1)
        return embedded


class Pointer_network(nn.Module):
    def __init__(self, embed_size, hidden_size, number_glimpses, tanh_exploration, u_tanh, attention_type,
                 use_cuda=USE_CUDA):
        super(Pointer_network, self).__init__()

        self.number_glimpses = number_glimpses
        self.use_cuda = use_cuda

        self.embedding = GraphEmbedding(2, embed_size, use_cuda=use_cuda)
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder_initial_g = nn.Parameter(torch.FloatTensor(embed_size))
        self.decoder_initial_g.data.uniform_(-(1. / math.sqrt(embed_size)), 1. / math.sqrt(embed_size))
        self.point_to_output = Attention(hidden_size, use_tanh=u_tanh, C=tanh_exploration, name=attention_type,
                                         use_cuda=use_cuda)
        self.glimpsing = Attention(hidden_size, use_tanh=False, name=attention_type, use_cuda=use_cuda)

    # function of glimpsing to aggregate the contribution of different part of the sequences
    def glimpse(self, qry, reference, mask, index):
        refer, logitt = self.glimpsing(qry, reference)
        logitt, masking = self.masking_logits(logitt, mask, index)
        query = torch.bmm(refer, F.softmax(logitt).unsqueeze(2)).squeeze(2)
        return query, masking

    def reward(self, tour, use_cuda='TRUE'):
        tour_len = Variable(torch.zeros([tour[0].size(0)]))
        if use_cuda:
            tour_len = tour_len.cuda()
        for i in range(len(tour) - 1):
            tour_len += torch.norm(tour[i + 1] - tour[i], dim=1)
        tour_len += torch.norm(tour[-1] - tour[0], dim=1)
        return tour_len

    def masking_logits(self, logit, mask, index):
        mask_temp = mask.clone()
        if index is not None:
            mask_temp[[i for i in range(logit.size(0))], index.data] = 1
            logit[mask_temp] = -np.inf
        return logit, mask_temp

    def forward(self, input):
        # initialization
        indexs = None
        prob_sequence = []
        indexs_sequence = []
        action_sequence = []
        prob_action_sequence = []
        batch_size = input.size(0)
        seq_len = input.size(2)
        input_data = input
        mask = torch.zeros(batch_size, seq_len).byte()  # a tensor with 0 in size batch_size,seq_len
        decoder_initial_g = self.decoder_initial_g.unsqueeze(0).repeat(batch_size, 1)

        if self.use_cuda:
            mask = mask.cuda()

        # starting process of encoding
        vectorized_data = self.embedding(input)
        encoder_hidden_states, (last_hidden_states, last_context) = self.encoder_lstm(vectorized_data)

        # starting process of decoding and pointing
        for i in range(seq_len):
            _, (last_hidden_states, last_context) = self.decoder_lstm(decoder_initial_g.unsqueeze(1),
                                                                      (last_hidden_states, last_context))
            current_hidden = last_hidden_states.squeeze(0)

            for i in range(self.number_glimpses):
                current_hidden, mask = self.glimpse(current_hidden, encoder_hidden_states, mask, indexs)

            _, output_logits = self.point_to_output(current_hidden, encoder_hidden_states)
            output_logits, mask = self.masking_logits(output_logits, mask, indexs)
            probs_of_current_output = F.softmax(output_logits)

            indexs = probs_of_current_output.multinomial(num_samples=1).squeeze(1)
            for indexsed in indexs_sequence:
                if indexsed.eq(indexs).data.any():
                    indexs = probs_of_current_output.multinomial(num_samples=1).squeeze(1)
                    break
            decoder_initial_g = vectorized_data[[i for i in range(batch_size)], indexs.data, :]

            prob_sequence.append(probs_of_current_output)
            indexs_sequence.append(indexs)

        action_data = input_data.transpose(1, 2)
        for id in indexs_sequence:
            action_sequence.append(action_data[[x for x in range(batch_size)], id.data, :])

        for prob, id in zip(prob_sequence, indexs_sequence):
            prob_action_sequence.append(prob[[x for x in range(batch_size)], id.data])

        Reward = self.reward(action_sequence)

        return Reward, prob_action_sequence, action_sequence, indexs_sequence



class Training:
    def __init__(self, Prt, data, v_data, alpha, batch_size=128, threshold=None):
        self.Prt = Prt
        self.alpha = alpha
        self.threshold = threshold
        self.train_data = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_data = DataLoader(v_data, batch_size=batch_size, shuffle=True, num_workers=0)
        self.Prt_opt = optim.Adam(Prt.parameters(), lr=1e-4)
        self.learning_factor = 0.96
        self.grad_norm = 2
        self.tour = []
        self.v_tour = []

    def training(self, epoch):
        critic_exp_mvg_avg = torch.zeros(1)
        if USE_CUDA:
            critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()
        for ep in range(epoch):
            for id_of_batch, batch_data in enumerate(self.train_data):
                self.Prt.train()
                input = Variable(batch_data)
                input = input.cuda()

                Reward, prob_action_sequence, action_sequence, indexs_sequence = self.Prt(input)
                if id_of_batch == 0:
                    critic_exp_mvg_avg = Reward.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * self.alpha) + ((1. - self.alpha) * Reward.mean())


                logprobs = 0
                for prob in prob_action_sequence:
                    logprob = torch.log(prob)
                    logprobs += logprob
                logprobs[(logprobs < -1000).detach()] = 0.
                # print(Reward.size()) 128
                # print(Bs.size())  128*1
                # print((Reward - Bs.mean()).size()) 128
                # print(logprobs.size())  128
                Batch_loss = (Reward - critic_exp_mvg_avg) * logprobs
                #   print(Batch_loss.size()) 128
                Prt_loss = Batch_loss.mean()
                self.Prt_opt.zero_grad()
                Prt_loss.backward()
                torch.nn.utils.clip_grad_norm(self.Prt.parameters(),
                                                  float(self.grad_norm), norm_type=2)
                self.Prt_opt.step()
                critic_exp_mvg_avg = critic_exp_mvg_avg.detach()
                self.tour.append(Reward.mean().data)


                if id_of_batch % 10 == 0:
                    self.plot()
                    #        if id_of_batch % 100 == 0:
                    self.validate(self.val_data)
             #   if self.threshold and (self.tour[-1] < self.threshold).__int__():
             #       print("EARLY STOPPAGE!")
             #   break

    def plot(self):
        clear_output(True)
        plt.figure(figsize=(30, 10))
        plt.subplot(131)
        plt.title('Current length of tour: %s' % (self.tour[-1] if len(self.tour) else 'initializing'))
        plt.plot(self.tour)
        plt.grid()
        plt.subplot(132)
        plt.title(
            'Current length of validated tour: %s' % (self.v_tour[-1] if len(self.v_tour) else 'initializing'))
        plt.plot(self.v_tour)
        plt.grid()
        plt.show()

    def validate(self, data):
        self.Prt.eval()
        for batch_sample in data:
            input = Variable(batch_sample)
            input = input.cuda()
            R, _, _, _ = self.Prt(input)
            self.v_tour.append(R.mean().data)

embed_size = 128
hidden_size = 128
numbers_of_glimpses = 1
Exploration_of_tanh = 10
use_tanh = True
alpha = 0.9

tsp_modeling = Pointer_network(embed_size, hidden_size, numbers_of_glimpses, Exploration_of_tanh, use_tanh,
                                   "Bahdanau")
if USE_CUDA:
    tsp_modeling = tsp_modeling.cuda()

Training_20 = Training(tsp_modeling, data_20, val_20, alpha, 128, threshold=3.99)
Training_20.training(8)



