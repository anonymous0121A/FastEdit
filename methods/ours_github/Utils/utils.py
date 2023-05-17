import torch as t
import torch.nn.functional as F

def cal_bpr(anc_embeds, pos_embeds, neg_embeds):
	pos_preds = (anc_embeds * pos_embeds).sum(-1)
	neg_preds = (anc_embeds * neg_embeds).sum(-1)
	return -((pos_preds - neg_preds).sigmoid() + 1e-10).log().mean()

def cal_reg(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret
