import torch
import torch.nn as nn

class DaConsistLoss(nn.Module):
    
    def __call__(self, ins_fea, img_feas, ins_labels, size_average=True):

        loss = []
        len_ins = ins_fea.size(0)
        intervals = [torch.nonzero(ins_labels).size(0), len_ins-torch.nonzero(ins_labels).size(0)]
        for img_fea_per_level in img_feas:
            N, A, H, W = img_fea_per_level.shape
            img_fea_per_level = torch.mean(img_fea_per_level.reshape(N, -1), 1)
            img_feas_per_level = []
            assert N==2, \
                "only batch size=2 is supported for consistency loss now, received batch size: {}".format(N)
            for i in range(N):
                img_fea_mean = img_fea_per_level[i].view(1, 1).repeat(intervals[i], 1)
                img_feas_per_level.append(img_fea_mean)
            img_feas_per_level = torch.cat(img_feas_per_level, dim=0)
            loss_per_level = torch.abs(img_feas_per_level - ins_fea)
            loss.append(loss_per_level)
        loss = torch.cat(loss, dim=1)
        if size_average:
            return loss.mean()

        return loss.sum()
