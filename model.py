import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from aggregator import LocalAggregator, GlobalAggregator
from torch.nn import Module
import datetime
from tqdm import tqdm


class GC_TAGNN(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(GC_TAGNN, self).__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()

        self.embedding = nn.Embedding(num_node, self.dim)
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)

        self.global_agg = []
        for i in range(self.hop):
            agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Target-aware attention layers
        self.linear_one = nn.Linear(self.dim, self.dim)
        self.linear_two = nn.Linear(self.dim, self.dim)
        self.linear_three = nn.Linear(self.dim, 1, bias=False)
        self.linear_transform = nn.Linear(self.dim * 2, self.dim)
        self.linear_t = nn.Linear(self.dim, self.dim)

        # Gated fusion: new
        self.linear_gate = nn.Linear(self.dim * 2, self.dim)
        self.sigmoid = nn.Sigmoid()

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        alpha = F.softmax(alpha, 1)
        s_global = torch.sum(alpha * hidden * mask.unsqueeze(-1).float(), 1)
        s_hybrid = self.linear_transform(torch.cat([s_global, ht], 1))

        b = self.embedding.weight[1:]
        masked_hidden = hidden * mask.unsqueeze(-1).float()
        qt = self.linear_t(masked_hidden)
        beta = F.softmax(b @ qt.transpose(1, 2), -1)
        s_target = beta @ masked_hidden

        final_session_representation = s_hybrid.unsqueeze(1) + s_target
        scores = torch.sum(final_session_representation * b, -1)
        return scores

    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)
        h_local = self.local_agg(h, adj, mask_item)

        # --- Global aggregation ---
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        sum_item_emb = sum_item_emb.unsqueeze(-2)

        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vector=entity_vectors[hop + 1].view(shape),
                    masks=None,
                    batch_size=batch_size,
                    neighbor_weight=weight_neighbors[hop].view(batch_size, -1, self.sample_num),
                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)

        # --- Gated fusion ---
        h_local = F.dropout(h_local, self.opt.dropout_local, training=self.training)
        h_global = F.dropout(h_global, self.opt.dropout_global, training=self.training)
        gate = self.sigmoid(self.linear_gate(torch.cat([h_local, h_global], -1)))
        output = gate * h_global + (1 - gate) * h_local
        return output


def trans_to_cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def trans_to_cpu(x):
    return x.cpu() if torch.cuda.is_available() else x


def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    hidden = model(items, adj, mask, inputs)
    seq_hidden = torch.stack([hidden[i][alias_inputs[i]] for i in range(len(alias_inputs))])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True, num_workers=4)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True, num_workers=4)
    result = []
    hit, mrr = [], []
    for data in test_loader:
        targets, scores = forward(model, data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)
    return result
