import datetime
import math
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F
import numpy as np


class GC_TAGNN(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(GC_TAGNN, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        
        # <<< START: ADDED FROM TAGNN >>>
        # این آرگومان برای سازگاری با منطق اصلی TAGNN اضافه شده است
        self.nonhybrid = opt.nonhybrid 
        # <<< END: ADDED FROM TAGNN >>>

        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item representation
        self.embedding = nn.Embedding(num_node, self.dim)

        # <<< START: LAYERS FOR TARGET ATTENTION (from TAGNN) >>>
        # این لایه‌ها مستقیماً از مدل TAGNN برای پیاده‌سازی مکانیزم‌های توجه اضافه شده‌اند
        self.linear_one = nn.Linear(self.dim, self.dim, bias=True)
        self.linear_two = nn.Linear(self.dim, self.dim, bias=True)
        self.linear_three = nn.Linear(self.dim, 1, bias=False)
        self.linear_transform = nn.Linear(self.dim * 2, self.dim, bias=True)
        # لایه کلیدی برای مکانیزم توجه به هدف
        self.linear_t = nn.Linear(self.dim, self.dim, bias=False)  
        # <<< END: LAYERS FOR TARGET ATTENTION >>>

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

    # --------------------------------------------------------------------------------
    # <<< THIS IS THE NEW COMPUTE_SCORES METHOD, ADAPTED FROM TAGNN >>>
    # --------------------------------------------------------------------------------
    def compute_scores(self, hidden, mask):
        """
        این متد به طور کامل با منطق امتیازدهی TAGNN جایگزین شده است.
        ورودی `hidden` همان بازنمایی آیتم‌هاست که از بخش GNN (ترکیب زمینه محلی و سراسری) به دست آمده است.
        """
        # گام ۱: محاسبه بردار جلسه عمومی (بدون توجه به هدف)
        # ht بازنمایی آخرین آیتم در جلسه است
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # (batch_size, hidden_size)
        
        # محاسبه وزن توجه برای ترکیب بازنمایی آیتم‌های جلسه
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # (batch_size, 1, hidden_size)
        q2 = self.linear_two(hidden)  # (batch_size, seq_length, hidden_size)
        alpha = self.linear_three(torch.sigmoid(q1 + q2)) # (batch_size, seq_length, 1)
        alpha = F.softmax(alpha, 1) # نرمال‌سازی وزن‌ها
        
        # s_global بردار وزنی از تمام آیتم‌های جلسه است
        s_global = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)  # (batch_size, hidden_size)
        
        # ترکیب بردار جلسه عمومی با بازنمایی آخرین آیتم
        if not self.nonhybrid:
            s_hybrid = self.linear_transform(torch.cat([s_global, ht], 1))
        else:
            s_hybrid = s_global

        # گام ۲: محاسبه بخش توجه به هدف (Target-Aware Attention)
        # b بازنمایی تمام آیتم‌های کاندید در دیتاست است
        b = self.embedding.weight[1:]  # (n_nodes, hidden_size)
        
        masked_hidden = hidden * mask.view(mask.shape[0], -1, 1).float()
        
        # اعمال ترنسفورمیشن خطی روی بازنمایی آیتم‌های جلسه برای محاسبه توجه به هدف
        qt = self.linear_t(masked_hidden)  # (batch_size, seq_length, hidden_size)
        
        # محاسبه امتیاز توجه برای هر آیتم کاندید نسبت به تمام آیتم‌های جلسه
        beta = F.softmax(b @ qt.transpose(1, 2), -1)  # (batch_size, n_nodes, seq_length)
        
        # s_target بردار جلسه وابسته به هدف است
        s_target = beta @ masked_hidden  # (batch_size, n_nodes, hidden_size)

        # گام ۳: ترکیب بردارها و محاسبه امتیاز نهایی
        s_hybrid_view = s_hybrid.view(hidden.shape[0], 1, hidden.shape[2]) # (batch_size, 1, hidden_size)
        
        # بردار نهایی جلسه از ترکیب بخش عمومی و بخش وابسته به هدف به دست می‌آید
        final_session_representation = s_hybrid_view + s_target # (batch_size, n_nodes, hidden_size)

        # محاسبه امتیاز نهایی با ضرب داخلی
        scores = torch.sum(final_session_representation * b, -1)  # (batch_size, n_nodes)
        
        return scores
    # --------------------------------------------------------------------------------
    # <<< END OF NEW METHOD >>>
    # --------------------------------------------------------------------------------

    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # Local graph aggregation (from GCE-GNN)
        h_local = self.local_agg(h, adj, mask_item)

        # Global graph aggregation (from GCE-GNN)
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

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
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop+1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)

        # Combine local and global representations
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        output = h_local + h_global

        return output


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()

    hidden = model(items, adj, mask, inputs)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
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
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
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