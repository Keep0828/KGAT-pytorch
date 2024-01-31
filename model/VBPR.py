import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class VBPR(nn.Module):

    def __init__(self, args,
                 n_users, n_items,
                 user_pre_embed=None, item_pre_embed=None,
                 item_visual_pre_embed=None):

        super(VBPR, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = args.embed_dim
        self.l2loss_lambda = args.l2loss_lambda

        self.user_embed = nn.Embedding(self.n_users, self.embed_dim)
        self.item_embed = nn.Embedding(self.n_items, self.embed_dim)

        if (self.use_pretrain == 1) and (user_pre_embed is not None):
            self.user_embed.weight = nn.Parameter(user_pre_embed)
        else:
            nn.init.xavier_uniform_(self.user_embed.weight)

        if (self.use_pretrain == 1) and (item_pre_embed is not None):
            self.item_embed.weight = nn.Parameter(item_pre_embed)
        else:
            nn.init.xavier_uniform_(self.item_embed.weight)

        self.use_visual_bias = False  # 是否加偏置项

        self.item_visual_feature = nn.Embedding.from_pretrained(item_visual_pre_embed, freeze=True)
        self.visual_trans = nn.Linear(self.item_visual_feature.weight.shape[1], self.embed_dim, bias=False)
        self.user_visual_embed = nn.Embedding(self.n_users, self.embed_dim)

        if self.use_visual_bias:
            self.visual_bias_term = nn.Embedding.from_pretrained(torch.Tensor(self.n_users, self.item_visual_feature.weight.shape[1]), freeze=False)

    def calc_score(self, user_ids, item_ids):
        """
        user_ids:   (n_users)
        item_ids:   (n_items)
        """
        user_embed = self.user_embed(user_ids)                              # (n_users, embed_dim)
        user_visual_embed = self.user_visual_embed(user_ids)
        item_embed = self.item_embed(item_ids)                              # (n_items, embed_dim)
        item_visual_embed = self.visual_trans(self.item_visual_feature(item_ids))

        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))     # (n_users, n_items)
        visual_score = torch.matmul(user_visual_embed, item_visual_embed.transpose(0, 1))
        if self.use_visual_bias:
            final_score = cf_score + visual_score + torch.matmul(self.visual_bias_term(user_ids), self.item_visual_feature(item_ids).transpose(0, 1))
        else:
            final_score = cf_score + visual_score
        return final_score


    def calc_loss(self, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (batch_size)
        item_pos_ids:   (batch_size)
        item_neg_ids:   (batch_size)
        """
        user_embed = self.user_embed(user_ids)              # (batch_size, embed_dim)
        item_pos_embed = self.item_embed(item_pos_ids)      # (batch_size, embed_dim)
        item_neg_embed = self.item_embed(item_neg_ids)      # (batch_size, embed_dim)

        pos_id_score = torch.sum(user_embed * item_pos_embed, dim=1)       # (batch_size)
        neg_id_score = torch.sum(user_embed * item_neg_embed, dim=1)       # (batch_size)

        user_visual_embed = self.user_visual_embed(user_ids)
        item_pos_visual_embed = self.visual_trans(self.item_visual_feature(item_pos_ids))
        item_neg_visual_embed = self.visual_trans(self.item_visual_feature(item_neg_ids))

        pos_visual_score = torch.sum(user_visual_embed * item_pos_visual_embed, dim=1)  # (batch_size)
        neg_visual_score = torch.sum(user_visual_embed * item_neg_visual_embed, dim=1)  # (batch_size)

        if self.use_visual_bias:
            user_bias_embed = self.visual_bias_term(user_ids)
            item_pos_bias_embed = self.item_visual_feature(item_pos_ids)
            item_neg_bias_embed = self.item_visual_feature(item_neg_ids)

            pos_bias_score = torch.sum(user_bias_embed * item_pos_bias_embed, dim=1)  # (batch_size)
            neg_bias_score = torch.sum(user_bias_embed * item_neg_bias_embed, dim=1)  # (batch_size)

            pos_score = pos_id_score + pos_visual_score + pos_bias_score
            neg_score = neg_id_score + neg_visual_score + neg_bias_score
        else:
            pos_score = pos_id_score + pos_visual_score
            neg_score = neg_id_score + neg_visual_score

        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.l2loss_lambda * l2_loss
        return loss


    def forward(self, *input, is_train):
        if is_train:
            return self.calc_loss(*input)
        else:
            return self.calc_score(*input)


