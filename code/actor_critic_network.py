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
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Concate', use_cuda=USE_CUDA):
        super(Attention, self).__init__()

        self.use_tanh = use_tanh
        self.C = C
        self.name = name

        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1, 1)

        V = torch.FloatTensor(hidden_size)
        if use_cuda:
            V = V.cuda()
        self.V = nn.Parameter(V)
        self.V.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

    def forward(self, query, ref):

        batch_size = ref.size(0)
        seq_len = ref.size(1)

        if self.name == 'Concate':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)
            ref = self.W_ref(ref)
            expanded_query = query.repeat(1, 1, seq_len)
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
            logits = torch.bmm(V, F.tanh(expanded_query + ref)).squeeze(1)
        else:
            raise NotImplementedError

        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits
        return ref, logits



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


class Critic_network(nn.Module):
    def __init__(self, embed_size, hidden_size, number_glimpses, attention_type, use_cuda=USE_CUDA):
        super(Critic_network, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_glimpses = number_glimpses
        self.use_cuda = use_cuda
        self.embedding = GraphEmbedding(2, embed_size, use_cuda=use_cuda)
        self.encoder_lstm_c = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.glimpsing = Attention(hidden_size, use_tanh=False, name=attention_type, use_cuda=use_cuda)

    def forward(self, input):
        vectorized_data = self.embedding(input)
        encoder_hidden_states, (last_hidden_states, last_context) = self.encoder_lstm_c(vectorized_data)
        current_hidden_states = last_hidden_states.squeeze(0)
        for i in range(self.number_glimpses):
            ref, logits = self.glimpsing(current_hidden_states, encoder_hidden_states)
            current_hidden_states = torch.bmm(ref, F.softmax(logits).unsqueeze(2)).squeeze(2)
        out = self.layer1(current_hidden_states)
        out = self.relu(out)
        out1 = self.layer2(out)
        return out1


class Training:
    def __init__(self, Prt, Crt, data, v_data, alpha, batch_size=128, threshold=None):
        self.Prt = Prt
        self.Crt = Crt
        self.alpha = alpha
        self.threshold = threshold
        self.train_data = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_data = DataLoader(v_data, batch_size=batch_size, shuffle=True, num_workers=0)
        self.Prt_opt = optim.Adam(Prt.parameters(), lr=1e-3)
        self.Crt_opt = optim.Adam(Crt.parameters(), lr=1e-3)
        self.learning_factor = 0.9
        self.grad_norm = 2
        self.tour = []
        self.v_tour = []
        self.lss = []
        self.decay_w=0

    def training(self, epoch):
        rate_ind = 0
        decay_ind = 0
        for ep in range(epoch):
            for id_of_batch, batch_data in enumerate(self.train_data):
                rate_ind += 1
                if rate_ind % 2000 == 0:
                    decay_ind += 1
                    self.decay_w = self.learning_factor ** decay_ind
                    self.Prt_opt = optim.Adam(self.Prt.parameters(), lr=(1e-3 * self.decay_w))
                    self.Crt_opt = optim.Adam(self.Crt.parameters(), lr=(1e-3 * self.decay_w))
                self.Prt.train()
                self.Crt.train()
                input = Variable(batch_data)
                input = input.cuda()

                Reward, prob_action_sequence, action_sequence, indexs_sequence = self.Prt(input)
                Bs = self.Crt(input)
                Bs.detach_()
                logprobs = 0
                for prob in prob_action_sequence:
                    logprob = torch.log(prob)
                    logprobs += logprob
                logprobs[(logprobs < -1000).detach()] = 0.
                # print(Reward.size()) 128
                # print(Bs.size())  128*1
                # print((Reward - Bs.mean()).size()) 128
                # print(logprobs.size())  128
                Batch_loss = (Reward - Bs.squeeze(1)) * logprobs
                #   print(Batch_loss.size()) 128
                Prt_loss = Batch_loss.mean()
                self.Prt_opt.zero_grad()
                Prt_loss.backward()
                torch.nn.utils.clip_grad_norm(self.Prt.parameters(),
                                                  float(self.grad_norm), norm_type=2)
                self.Prt_opt.step()
                Reward.detach_()

                Bs_reward = self.Crt(input)
                tt = Bs_reward - Reward.unsqueeze(1)
                Batch_c_loss = tt ** 2
                Crt_loss = Batch_c_loss.mean()
                self.Crt_opt.zero_grad()
                Crt_loss.backward()
                torch.nn.utils.clip_grad_norm(self.Crt.parameters(),
                                                  float(self.grad_norm), norm_type=2)
                self.Crt_opt.step()
                self.tour.append(Reward.mean().data)
                if Crt_loss.data.__float__() < 3:
                    self.lss.append(Crt_loss.data)

                if id_of_batch % 10 == 0:
                    self.plot()
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
        plt.subplot(133)
        plt.title('Current value of loss of baseline estimation,starting below 3')
        plt.plot(self.lss)
        plt.grid()
        plt.show()

    def validate(self, data):
        self.Prt.eval()
        for batch_sample in data:
            input = Variable(batch_sample)
            input = input.cuda()
            R, _, _, _ = self.Prt(input)
        self.v_tour.append(R.mean().data)

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

embed_size = 128
hidden_size = 128
numbers_of_glimpses = 1
Exploration_of_tanh = 10
use_tanh = True
alpha = 0.9

tsp_modeling = Pointer_network(embed_size, hidden_size, numbers_of_glimpses, Exploration_of_tanh, use_tanh,
                                   "Concate")
critic_tsp = Critic_network(embed_size, hidden_size, number_glimpses=3, attention_type="Concate")
if USE_CUDA:
    tsp_modeling = tsp_modeling.cuda()
    critic_tsp = critic_tsp.cuda()

Training_20 = Training(tsp_modeling, critic_tsp, data_20, val_20, alpha, 128, threshold=3.99)
Training_20.training(8)


