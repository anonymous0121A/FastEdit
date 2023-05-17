import torch as t
import torch.nn.functional as F

def cal_bpr(anc_embeds, pos_embeds, neg_embeds):
	pos_preds = (anc_embeds * pos_embeds).sum(-1)
	neg_preds = (anc_embeds * neg_embeds).sum(-1)
	# print('loss', (pos_preds - neg_preds).sigmoid().log())
	# print('sigmoid', (pos_preds - neg_preds).sigmoid())
	return -((pos_preds - neg_preds).sigmoid() + 1e-10).log().mean()

def _crr_neg(embeds1, embeds2, temp):
	embeds1 = F.normalize(embeds1)
	embeds2 = F.normalize(embeds2)
	tem = embeds1 @ embeds2.T
	return t.log(t.exp((tem) / temp).sum(-1) + 1e-10).mean()

def cal_crr(usr_embeds, itm_embeds, ancs, poss, temp, inbatch=False):
	pos_term = 0#-(((usr_embeds[ancs] * itm_embeds[poss]).sum(-1))).mean()
	# pos_term = -(usr_embeds[ancs] * itm_embeds[poss]).sum(-1).mean()
	if not inbatch:
		neg_term = _crr_neg(usr_embeds[ancs], usr_embeds, temp) + _crr_neg(itm_embeds[poss], itm_embeds, temp) + _crr_neg(usr_embeds[ancs], itm_embeds, temp)
	else:
		neg_term = _crr_neg(usr_embeds[ancs], usr_embeds[ancs], temp) + _crr_neg(itm_embeds[poss], itm_embeds[poss], temp) + _crr_neg(usr_embeds[ancs], itm_embeds[poss], temp)
	return pos_term + neg_term

def cal_reg(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret
