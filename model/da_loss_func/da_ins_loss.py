import torch
import torch.nn as nn
import torch.nn.functional as F


class DaInsLoss(nn.Module):
    
    def __call__(self, da_ins_features, da_ins_labels):

        da_ins_loss = F.binary_cross_entropy_with_logits(
            torch.squeeze(da_ins_features), torch.squeeze(da_ins_labels.type(torch.FloatTensor))
        )

        return da_ins_loss
