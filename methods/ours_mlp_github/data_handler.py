import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
from Utils.time_logger import log
import torch as t
import torch.utils.data as data

class DataHandler:
    def __init__(self):
        if args.data == 'ml1m':
            predir = '/home/___/data/unlearning datasets/ml-1m' + '/'
        elif args.data == 'ml10m':
            predir = '/home/___/data/unlearning datasets/ml-10m/'
        elif args.data == 'yelp':
            predir = '/home/___/data/unlearning datasets/yelp2018/'
        self.trn_file = predir + 'trn_mat.pkl'
        self.tst_file = predir + 'tst_mat.pkl'

    def _load_one_file(self, filename, non_binary=False):
        with open(filename, 'rb') as fs:
            tem = pickle.load(fs)
            ret = tem if non_binary else (tem != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def _normalize_adj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        if mat.shape[0] == mat.shape[1]:
            return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
        else:
            tem = d_inv_sqrt_mat.dot(mat)
            col_degree = np.array(mat.sum(axis=0))
            d_inv_sqrt = np.reshape(np.power(col_degree, -0.5), [-1])
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
            return tem.dot(d_inv_sqrt_mat).tocoo()
    
    def _scipy_to_torch_adj(self, mat):
        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()
    
    def _make_torch_adj(self, mat, self_loop=False):
        # make ui adj
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        bi_mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        bi_mat = (bi_mat != 0) * 1.0
        if self_loop:
            bi_mat = (bi_mat + sp.eye(bi_mat.shape[0])) * 1.0
        bi_mat = self._normalize_adj(bi_mat)
        uni_mat = self._normalize_adj(mat)
        return self._scipy_to_torch_adj(uni_mat), self._scipy_to_torch_adj(bi_mat)
    
    def random_drop_edges(self, rate, ini_drop=False):
        rows = self.trn_mat.row
        cols = self.trn_mat.col
        vals = self.trn_mat.data
        length = rows.shape[0]
        pick = np.random.permutation(length)[:int((1-rate)*length)]
        rows = rows[pick]
        cols = cols[pick]
        vals = vals[pick]
        dropped_mat = coo_matrix((vals, (rows, cols)), shape=self.trn_mat.shape)
        if ini_drop:
            return dropped_mat
        else:
            return self._make_torch_adj(dropped_mat)

    def load_data(self, drop_rate=0.0):
        trn_mat = self._load_one_file(self.trn_file)
        tst_mat = self._load_one_file(self.tst_file)

        if drop_rate > 0:
            self.trn_mat = trn_mat
            trn_mat = self.random_drop_edges(drop_rate, True)

        self.trn_mat = trn_mat
        args.user, args.item = trn_mat.shape
        self.torch_uni_adj, self.torch_adj = self._make_torch_adj(trn_mat)

        trn_data = TrnData(trn_mat)
        self.trn_loader = data.DataLoader(trn_data, batch_size=args.batch, shuffle=True, num_workers=0)

        tst_data = TstData(tst_mat, trn_mat)
        self.tst_loader = data.DataLoader(tst_data, batch_size=args.tst_bat, shuffle=False, num_workers=0)

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(args.item)
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
    def __init__(self, coomat, trn_mat):
        self.csrmat = (trn_mat.tocsr() != 0) * 1.0

        tst_locs = [None] * coomat.shape[0]
        tst_usrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tst_locs[row] is None:
                tst_locs[row] = list()
            tst_locs[row].append(col)
            tst_usrs.add(row)
        tst_usrs = np.array(list(tst_usrs))
        self.tst_usrs = tst_usrs
        self.tst_locs = tst_locs

    def __len__(self):
        return len(self.tst_usrs)

    def __getitem__(self, idx):
        return self.tst_usrs[idx], np.reshape(self.csrmat[self.tst_usrs[idx]].toarray(), [-1])

class temHandler:
    def __init__(self, adjs):
        self.torch_uni_adj, self.torch_adj = adjs
