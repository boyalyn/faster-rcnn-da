import torch
import torch.nn as nn
import torch.nn.functional as F


class DaImgLoss(nn.Module):

    def __call__(self, da_img, targets):

        masks = self.prepare_masks(targets)
        masks = torch.cat(masks, dim=0)

        da_img_flattened = []
        da_img_labels_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment
        for da_img_per_level in da_img:
            N, A, H, W = da_img_per_level.shape
            # 2,1,38,76
            da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
            da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
            da_img_label_per_level[masks, :] = 1

            da_img_per_level = da_img_per_level.reshape(N, -1)
            da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
            
            da_img_flattened.append(da_img_per_level)
            da_img_labels_flattened.append(da_img_label_per_level)
            
        da_img_flattened = torch.cat(da_img_flattened, dim=0)
        da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=0)
#         print(da_img_labels_flattened.type(torch.cuda.BoolTensor))
        da_img_loss = F.binary_cross_entropy_with_logits(
            da_img_flattened, da_img_labels_flattened
        )

        return da_img_loss

    def prepare_masks(self, targets):
        masks = []
        for targets_per_image in targets: 
            # this is official code, but doesn't work here
            # is_source = targets_per_image.get_field('is_source')
            is_source = targets_per_image
            mask_per_image = is_source.new_ones(1, dtype=torch.bool) if is_source.any() else is_source.new_zeros(1, dtype=torch.bool)
            masks.append(mask_per_image)
        return masks 
    

if __name__ == "__main__":
    
    # this is what images should be like
    dummy_img = [torch.rand((2,1,38,76))]
    # this is what labels should be like
    dummy_target = torch.zeros(2)
    loss = DaImgLoss()
    print(loss(dummy_img,dummy_target))