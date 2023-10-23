import sys
sys.path.append("/Users/boyaliu/Projects/simple-faster-rcnn")
import torch
from model.faster_rcnn import FasterRCNN
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn_vgg16 import VGG16RoIHead, decom_vgg16
from model.da_components import DAHead
from model.da_loss_func.da_consist_loss import DaConsistLoss
from model.da_loss_func.da_img_loss import DaImgLoss
from model.da_loss_func.da_ins_loss import DaInsLoss
from model.utils.bbox_tools import bbox2loc, loc2bbox
from Utils import array_tool as at
from Configs.Config import Config
# torch.set_default_datatype(torch.float32)

class DANN(FasterRCNN):

    def __init__(self,
                 n_fg_class=2,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        extractor, classifier = decom_vgg16()
        self.feat_stride = 16

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class+1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(DANN, self).__init__(
            extractor,
            rpn,
            head,
        )

        self.da_head = DAHead(512, 512*7*7)

        self.ins_loss = DaInsLoss()
        self.img_loss = DaImgLoss()
        self.cons_loss = DaConsistLoss()

    
    def da_forward(self, h, rois, roi_indices, domain_label, scale=1., ratios=None):

        # img_size = x.shape[2:]

        # h = self.extractor(x)

        # rpn_locs, rpn_scores, rois, roi_indices, anchor = \
        #     self.rpn(h, img_size, scale)

        roi_cls_locs, roi_scores, fc7 = self.head(
            h, rois, roi_indices, return_latent=True)

        # ratios = ratios.reshape(1,-1).to(Config().device)
        # da_img_features, da_img_consist_features, da_ins_features, da_ins_center, da_ins_consist_features = self.da_head(fc7, [h])
        da_outputs = self.da_head(fc7,[h]) # caution !!!!!!!!!!!!!!
        da_img_features, da_img_consist_features, da_ins_features, da_ins_center, da_ins_consist_features = da_outputs

        # create domain labels
        # da_img_label = []
        roi_cls_loc = roi_cls_locs.data.view(-1,self.n_class,4)
        roi = at.totensor(rois) / scale
        roi = roi.view((-1, 1, 4)).expand_as(roi_cls_loc)
        # cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
        #             at.tonumpy(roi_cls_loc).reshape((-1, 4)))

        locs = roi + roi_cls_loc
        
        if domain_label == "source":
            da_ins_label = torch.zeros_like(da_ins_features)
            # da_img_label = torch.zeros_like(da_img_features[0][:,:,])
            da_img_label = torch.zeros(da_img_features[0].shape[0])
        else:
            da_ins_label = torch.ones_like(da_ins_features)
            # da_img_label = torch.ones_like(da_img_features)
            da_img_label = torch.ones(da_img_features[0].shape[0])

        da_ins_label = da_ins_label.to(Config().device)
        da_img_label = da_img_label.to(Config().device)

        losses = self.compute_losses(da_outputs, da_ins_label, da_img_label)

        # print(losses) 

        # return roi_cls_locs, roi_scores, rois, roi_indices
        return losses
    

    def compute_losses(self, da_outputs, da_ins_labels, da_img_labels):

        da_img_features, da_img_consist_features, da_ins_features, da_ins_center, da_ins_consist_features = da_outputs

        """ da ins loss """
        da_ins_loss = self.ins_loss(da_ins_features, da_ins_labels)

        """ da img loss """
        da_img_loss = self.img_loss(da_img_features, da_img_labels)

        """ da cons loss """
        # da_cons_loss = self.cons_loss(da_ins_features, da_img_features, da_ins_labels)

        alpha, beta, gamma = 0.4, 0.4, 0.2
        
        # return alpha*da_ins_loss, beta*da_img_loss, gamma*da_cons_loss
        return alpha*da_ins_loss, beta*da_img_loss

        

if  __name__ == "__main__":

    input_tensor = torch.rand((2,3,512,512))
    domain_label = 'source'
    model = DANN()

    model(input_tensor,domain_label)