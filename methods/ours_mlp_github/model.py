from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from params import args
from Utils.utils import *
import numpy as np
import scipy
from data_handler import temHandler

init = nn.init.xavier_uniform_
uniform_init = nn.init.uniform_

class UntrainedGCN(nn.Module):
    def __init__(self, adjs):
        super(UntrainedGCN, self).__init__()
        uni_adj, bi_adj = adjs
        self.adj = bi_adj

        self.uni_adj = uni_adj
        self._make_ini_embeds()

    def _torch_to_scipy(self, torch_adj):
        idxs = torch_adj._indices().detach().cpu().numpy()
        rows, cols = idxs[0, :], idxs[1, :]
        data = torch_adj._values().detach().cpu().numpy()
        return scipy.sparse.csr_matrix((data, (rows, cols)), shape=[args.user, args.item])
    
    def _make_ini_embeds(self):
        if args.ini_embed == 'uniform':
            self.ini_embeds = init(t.empty(args.user + args.item, args.latdim)).detach().cuda()
        elif args.ini_embed == 'svd':
            q = args.latdim // 2 if args.concat==1 else args.latdim
            svd_u, s, svd_v = t.svd_lowrank(self.uni_adj, q=q, niter=args.niter)
            svd_u = svd_u @ t.diag(t.sqrt(s))
            svd_v = svd_v @ t.diag(t.sqrt(s))
            self.ini_embeds = t.concat([svd_u, svd_v], dim=0)
        else:
            raise Exception('Unrecognized Initial Embedding')

    def forward(self, adj=None, return_split=True):
        if adj is None:
            adj = self.adj
        embeds_list = [self.ini_embeds]
        for i in range(args.gnn_layer):
            embeds = t.spmm(adj, embeds_list[-1])
            embeds_list.append(embeds)
        
        if args.concat == 1:
            tem_embeds = t.concat(embeds_list[-2:], dim=-1)
            usr_embeds = tem_embeds[:args.user]
            itm_embeds = tem_embeds[args.user:]
        else:
            usr_embeds = embeds_list[-1][:args.user]
            pck_order = -1 if args.symetric else -2
            itm_embeds = embeds_list[pck_order][args.user:]

        if not return_split:
            return t.concat([usr_embeds, itm_embeds], dim=0)
        return usr_embeds, itm_embeds

    def full_predict(self, usrs, trn_mask):
        usr_embeds, itm_embeds = self.forward()
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds

class UnlearningMLP(nn.Module):
    def __init__(self, handler):
        super(UnlearningMLP, self).__init__()
        self.topo_encoder = None
        self.set_topo_encoder(handler)
        self.layers = nn.Sequential(*[FeedForwardLayer(args.latdim, args.latdim, act=args.act) for i in range(args.layer_mlp)])
        self.layer_norm = nn.LayerNorm(args.latdim)
    
    def set_topo_encoder(self, handler):
        if args.ini_embed == 'uniform' and self.topo_encoder is not None:
            uni_adj = handler.torch_uni_adj
            adj = handler.torch_adj
            self.topo_encoder.adj = adj
            self.topo_encoder.uni_adj = uni_adj
        elif args.ini_embed == 'svd' or self.topo_encoder is None:
            adjs = (handler.torch_uni_adj, handler.torch_adj)
            self.topo_encoder = UntrainedGCN(adjs)
        else:
            raise Exception()
        self.adj = handler.torch_adj
        self.is_training = True
    
    
    def unlearn(self, adjs):
        tem_handler = temHandler(adjs)
        self.set_topo_encoder(tem_handler)

    def forward(self, adj=None, edges=None):
        if not self.is_training:
            embeds = self.final_embeds
            return embeds[:args.user], embeds[args.user:]
        if adj is None:
            embeds = self.topo_encoder(return_split=False)
        else:
            adj = self._mask_edges(adj, edges)
            embeds = self.topo_encoder(adj, return_split=False)
        for i, layer in enumerate(self.layers):
            embeds = layer(embeds)
        embeds = self.layer_norm(embeds)
        self.final_embeds = embeds
        return embeds[:args.user], embeds[args.user:]

    def cal_loss(self, batch_data):
        self.is_training = True
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        usr_embeds, itm_embeds = self.forward(self.adj, (ancs, poss))
        bpr_loss = cal_bpr(usr_embeds[ancs], itm_embeds[poss], itm_embeds[negs])
        reg_loss = cal_reg(self) * args.reg
        loss = bpr_loss + reg_loss
        loss_dict = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, loss_dict
    
    def full_predict(self, usrs, trn_mask):
        usr_embeds, itm_embeds = self.forward()
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        self.is_training = False
        return full_preds

    def _normalize_adj(self, adj):
        row_degree = t.pow(t.sparse.sum(adj, dim=1).to_dense(), 0.5)
        col_degree = t.pow(t.sparse.sum(adj, dim=0).to_dense(), 0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = row_degree[newRows], col_degree[newCols]
        newVals = adj._values() / rowNorm / colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)
    
    def _mask_edges(self, adj, edges):
        if args.mask is None or args.mask == 'none':
            return adj
        elif args.mask == 'trn':
            rows = adj._indices()[0, :]
            cols = adj._indices()[1, :]
            node_num = args.user + args.item
            hashvals = rows * node_num + cols

            pck_rows, pck_cols = edges
            pck_cols = pck_cols + args.user
            pck_hashvals1 = pck_rows * node_num + pck_cols
            pck_hashvals2 = pck_cols * node_num + pck_rows
            pck_hashvals = t.concat([pck_hashvals1, pck_hashvals2])
            
            for i in range(args.batch * 2 // args.mask_bat):
                hashvals = self._mask_edges_help(hashvals, pck_hashvals[i * args.mask_bat: (i+1) * args.mask_bat])

            cols = hashvals % node_num
            rows = t.div((hashvals - cols).long(), node_num, rounding_mode='trunc').long()
            adj = t.sparse.FloatTensor(t.stack([rows, cols], dim=0), t.ones_like(rows).cuda().float(), adj.shape)
            return self._normalize_adj(adj)
        elif args.mask == 'random':
            return self._random_mask_edge(adj)
    
    def _mask_edges_help(self, hashvals, pck_hashvals):
        idct = (hashvals.view([-1, 1]) - pck_hashvals.view([1, -1]) == 0).sum(-1).bool()
        hashvals = hashvals[t.logical_not(idct)]
        return hashvals

    def _random_mask_edge(self, adj):
        if args.keep_rate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + args.keep_rate).floor()).type(t.bool)
        newIdxs = idxs[:, mask]
        newVals = t.ones(newIdxs.shape[1]).cuda().float()
        return self._normalize_adj(t.sparse.FloatTensor(newIdxs, newVals, adj.shape))
    
class FeedForwardLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=True, act=None):
        super(FeedForwardLayer, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        if act == 'identity' or act is None:
            self.act = None
        elif act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.leaky)
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        else:
            raise Exception('Error')
    
    def forward(self, embeds):
        if self.act is None:
            return self.linear(embeds)
        return (self.act(self.linear(embeds))) + embeds
