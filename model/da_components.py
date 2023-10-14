import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.gradient_reverse import GradientScalarLayer
torch.set_default_datatype(torch.float32)


class DAHead(nn.Module):

    def __init__(self, in_channels, num_ins_inputs):

        super().__init__()
        self.resnet_backbone = False
        # reverse gradient scaling for img-level features
        self.grl_img = GradientScalarLayer(-1.0)
        # reverse gradient scaling for ins-level features
        self.grl_ins = GradientScalarLayer(-1.0)
        # gradient scaling for img-level features
        self.grl_img_consist = GradientScalarLayer(1.0)
        # gradient scaling for ins-level features
        self.grl_ins_consist = GradientScalarLayer(1.0)

        # instance level discriminator
        self.inshead = DAInsHead(num_ins_inputs)
        # image level discriminator
        self.imghead = DAImgHead(in_channels)
    
    def forward(self, da_ins_feature, img_features):

        # ???
        if self.resnet_backbone:
            da_ins_feature = self.avgpool(da_ins_feature)
        # expand instance-level features to (num_instance,-1)
        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)
        # pass hidden features to gradient scaling layers
        # image-level
        img_grl_fea = [self.grl_img(fea) for fea in img_features]
        img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]
        # instance-level
        ins_grl_fea = self.grl_ins(da_ins_feature)
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)
        # pass hidden features to domain discriminators
        # image-level
        da_img_features = self.imghead(img_grl_fea)
        da_img_consist_features = self.imghead(img_grl_consist_fea)
        da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
        # instance-level
        da_ins_features, da_ins_center = self.inshead(ins_grl_fea)
        da_ins_consist_features,_ = self.inshead(ins_grl_consist_fea)
        da_ins_consist_features = da_ins_consist_features.sigmoid()

        return da_img_features, da_img_consist_features, da_ins_features, da_ins_center, da_ins_consist_features


class DAImgHead(nn.Module):

    def __init__(self, in_channels):

        super(DAImgHead, self).__init__()
        # conv layers
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1,padding=1)
        self.conv2_da = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1,padding=1)
        self.conv3_da = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        # initialize conv layers
        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        
        # x is a list containing multi-scale image features
        img_features = []
        # collect features from each level
        for feature in x:
            t = F.relu(self.conv1_da(feature),inplace=True)
            t = F.relu(self.conv2_da(t),inplace=True)
            img_features.append(self.conv3_da(t))

        return img_features



class DAInsHead(nn.Module):

    def __init__(self, in_channels):

        super(DAInsHead, self).__init__()
        # fully-connected layers
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        # initialize fully-connected layers
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):

        x = F.relu(self.fc1_da(x),inplace=True)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x),inplace=True)
        x = F.dropout(x, p=0.5, training=self.training) #(256, 1024)

        x1 = self.fc3_da(x)

        return x1, x