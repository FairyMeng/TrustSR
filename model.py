from torch import nn
import torch
import h5py
import numpy as np
import torch.nn.functional as F

def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)


class TrustSR(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, classes, views, classifier_dims, num_layers=1, final_act='tanh',
                 dropout_hidden=.5, dropout_input=0, batch_size=50, embedding_dim=-1, use_cuda=False ,lambda_epochs=1):
        super(TrustSR, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.onehot_buffer = self.init_emb()
        self.h2o = nn.Linear(hidden_size, output_size)
        self.create_final_activation(final_act)
        if self.embedding_dim == -1:
            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        else:
            self.look_up = nn.Embedding(input_size,embedding_dim)
            self.all_gru = nn.GRU(self.embedding_dim*3, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
            self.mlp = nn.Linear(self.embedding_dim, self.embedding_dim)
            self.id_gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
            self.info_gru = nn.GRU(self.embedding_dim*2, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        self.views = views
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])
        self = self.to(self.device)

    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))


    def idToimageTensor(self, input):
        with h5py.File('./datasets/Amazon_Toys_and_Games/image_tensor_data.hdf5', 'r') as f:
            id_list=[]
            for i in range(len(input)):
                str1 = str(input[i].item())
                a = f[str1]
                list1 = a[:]
                id_list.append(list1)
            tensor=torch.Tensor(np.array(id_list))
            embedded = tensor.unsqueeze(0)
            f.close()
        return embedded
        
    def idTotextTensor(self, input):
        with h5py.File('./datasets/Amazon_Toys_and_Games/text_tensor_data.hdf5', 'r') as f:
            id_list=[]
            for i in range(len(input)):
                str1 = str(input[i].item())
                a = f[str1]
                list1 = a[:]
                id_list.append(list1)
            tensor=torch.Tensor(np.array(id_list))
            embedded = tensor.unsqueeze(0)
            f.close()
        return embedded
        
    def DS_Combin(self, alpha):

        def DS_Combin_two(alpha1, alpha2):

            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            S_a = self.classes / u_a
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    def forward(self, input, hidden, target, global_step):
        X = dict()
        single_alpha = dict()
        single_alpha_a = dict()
        alpha_a = dict()
        alpha_a_list = []
        if self.embedding_dim == -1:
            embedded = self.onehot_encode(input)
            if self.training and self.dropout_input > 0:
                embedded = self.embedding_dropout(embedded)
            embedded = embedded.unsqueeze(0)
            output, hidden = self.gru(embedded, hidden)
            output = output.view(-1, output.size(-1))

            X[0] = output
            evidence = self.infer(X)
            loss = 0
            alpha = dict()
            for v_num in range(len(X)):
                alpha[v_num] = evidence[v_num] + 1
                loss += ce_loss(target, alpha[v_num], self.classes, global_step, self.lambda_epochs)
            alpha_a = alpha[0]

            evidence_a = alpha_a - 1
            loss += ce_loss(target, alpha_a, self.classes, global_step, self.lambda_epochs)
            loss = torch.mean(loss)
            return evidence, evidence_a, loss

        else:
            id_embedded = input.unsqueeze(0)
            id_embedded = self.look_up(id_embedded)
 
            image_embedded = self.idToimageTensor(input)
            image_embedded = image_embedded.to(self.device)
            text_embedded = self.idTotextTensor(input)
            text_embedded = text_embedded.to(self.device)

            all_embedded = torch.cat((image_embedded,text_embedded,id_embedded),dim=2)
            output, hidden = self.all_gru(all_embedded, hidden)
            output = output.view(-1, output.size(-1))
            X[0] = output
            evidence = self.infer(X)
            loss = 0
            alpha = dict()
            for v_num in range(len(X)):
                alpha[v_num] = evidence[v_num] + 1
                loss += ce_loss(target, alpha[v_num], self.classes, global_step, self.lambda_epochs)
            alpha_a = alpha[0]

            evidence_a = alpha_a - 1
            loss += ce_loss(target, alpha_a, self.classes, global_step, self.lambda_epochs)
            loss = torch.mean(loss)
            return evidence, evidence_a, loss
        

    def infer(self, input):
            evidence = dict()
            for v_num in range(self.views):
                evidence[v_num] = self.Classifiers[v_num](input[v_num])
            return evidence



    def init_emb(self):
        onehot_buffer = torch.FloatTensor(self.batch_size, self.output_size)
        onehot_buffer = onehot_buffer.to(self.device)
        return onehot_buffer

    def onehot_encode(self, input):
        self.onehot_buffer.zero_()
        index = input.view(-1, 1)
        one_hot = self.onehot_buffer.scatter_(1, index, 1)
        return one_hot

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.dropout_input)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)
        mask = mask.to(self.device)
        input = input * mask
        return input

    def init_hidden(self):
        try:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        except:
            self.device = 'cpu'
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0
    

class Classifier(nn.Module):
    def __init__(self, classifier_dims, classes):
        super(Classifier, self).__init__()
        self.num_layers = len(classifier_dims)
        self.fc = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i+1]))
        self.fc.append(nn.Linear(classifier_dims[self.num_layers-1], classes))
        self.fc.append(nn.Softplus())

    def forward(self, x):
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h