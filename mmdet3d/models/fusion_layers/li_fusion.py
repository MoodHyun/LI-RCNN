# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.functional import grid_sample

from mmdet3d.core.bbox.structures import (get_proj_mat_by_coord_type,
                                          points_cam2img)
from ..builder import FUSION_LAYERS
from . import apply_3d_transformation



@FUSION_LAYERS.register_module()
class LIFusion(BaseModule):
    """Fuse image features from multi-scale features.

    Args:
        img_channels (list[int] | int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels (int): Channels of point features
        mid_channels (int): Channels of middle layers
        out_channels (int): Channels of output fused features
        img_levels (int, optional): Number of image levels. Defaults to 3.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
            Defaults to 'LIDAR'.
        conv_cfg (dict, optional): Dict config of conv layers of middle
            layers. Defaults to None.
        norm_cfg (dict, optional): Dict config of norm layers of middle
            layers. Defaults to None.
        act_cfg (dict, optional): Dict config of activatation layers.
            Defaults to None.
        activate_out (bool, optional): Whether to apply relu activation
            to output features. Defaults to True.
        fuse_out (bool, optional): Whether apply conv layer to the fused
            features. Defaults to False.
        dropout_ratio (int, float, optional): Dropout ratio of image
            features to prevent overfitting. Defaults to 0.
        aligned (bool, optional): Whether apply aligned feature fusion.
            Defaults to True.
        align_corners (bool, optional): Whether to align corner when
            sampling features according to points. Defaults to True.
        padding_mode (str, optional): Mode used to pad the features of
            points that do not have corresponding image features.
            Defaults to 'zeros'.
        lateral_conv (bool, optional): Whether to apply lateral convs
            to image features. Defaults to True.
    """

    def __init__(self,
                 img_block,
                 fu_conv,
                 cross_conv,
                 de_conv,
                 img_channels,
                 pts_channels,
                 de_conv_kernels ,
                 de_conv_reduces,
                 I2P_path=None,
                 P2I_path=None,
                 init_cfg=None):
        super(LIFusion, self).__init__(init_cfg=init_cfg)

        self.img_block = None
        self.fu_conv = fu_conv
        self.cross_conv = cross_conv
        self.de_conv = de_conv
        self.I2P_path = I2P_path
        self.P2I_path = P2I_path
        self.img_channels = img_channels
        self.pts_channels = pts_channels
        self.de_conv_kernels = de_conv_kernels
        self.de_conv_reduces = de_conv_reduces
        # if init_cfg.CROSS_FUSION:
        #     self.Cross_Fusion = nn.ModuleList()
        # if init_cfg.USE_IM_DEPTH:
        #     self.IMG_CHANNELS[0] = self.IMG_CHANNELS[0] + 1
        #
        # if init_cfg.INPUT_CROSS_FUSION:
        #     self.IMG_CHANNELS[0] = self.IMG_CHANNELS[0] + 4



        if img_block:
            self.img_block = nn.ModuleList()
            for i in range(len(img_channels)):
                self.img_block.append(
                    nn.Sequential(
                        ConvModule(
                            in_channels=img_channels[i],
                            out_channels=img_channels[i + 1],
                            conv_cfg=dict(type='Conv2d'),
                            norm_cfg=dict(type='BN2d'),
                            stride=1),
                        # nn.Conv2d(img_channels[i],
                        #           img_channels[i + 1],
                        #           kernel_size = 3,
                        #           stride = 1,
                        #           padding = 1,
                        #           bias = False),
                        # nn.BatchNorm2d(),
                        # nn.ReLU(inplace = True),
                        nn.Conv2d(
                            in_channels=img_channels[i + 1],
                            out_channels=img_channels[i + 1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False)))
                if self.fu_conv:
                    self.fu_conv = nn.ModuleList()
                    if self.I2P_path:
                        self.fu_conv.append(
                            Atten_Fusion_Conv(img_channels[i + 1],
                                              pts_channels[i],
                                              pts_channels[i]))
                    else:
                        self.fu_conv.append(
                            Fusion_Conv(img_channels[i + 1] + pts_channels[i],
                                        pts_channels[i]))

                if self.cross_conv:
                    self.cross_conv = nn.ModuleList()
                    if self.P2I_path:
                        self.cross_conv.append(
                            Fusion_Cross_Conv_Gate(img_channels[i + 1], pts_channels[i],
                                                   img_channels[i + 1]))
                    else:
                        self.cross_conv.append(
                            Fusion_Cross_Conv(img_channels[i + 1] + pts_channels[i],
                                              img_channels[i + 1]))
                self.de_conv = nn.ModuleList()
                self.de_conv.append(nn.ConvTranspose2d(img_channels[i + 1], self.de_conv_reduces[i],
                                                      kernel_size=self.de_conv_kernels[i],
                                                      stride=self.de_conv_kenrels[i]))

        self.image_fusion=ConvModule(sum(self.de_conv_reduces), self.img_features_channels // 4, kernel_size=1),



    #     if self.ADD_Image_Attention:
    #         self.final_fusion_img_point = Atten_Fusion_Conv(self.IMG_FEATURES_CHANNEL // 4,
    #                                                         self.IMG_FEATURES_CHANNEL,
    #                                                         self.IMG_FEATURES_CHANNEL)
    #     else:
    #         self.final_fusion_img_point = Fusion_Conv(
    #             self.IMG_FEATURES_CHANNEL + self.IMG_FEATURES_CHANNEL // 4,
    #             self.IMG_FEATURES_CHANNEL)
    #
    # if init_cfg.USE_SELF_ATTENTION:
    #     self.context_conv3 = PointContext3D(init_cfg.RPN.SA_CONFIG,
    #                                         IN_DIM=init_cfg.RPN.SA_CONFIG.MLPS[2][0][-1] + init_cfg.RPN.SA_CONFIG.MLPS[2][1][
    #                                             -1])
    #     self.context_conv4 = PointContext3D(init_cfg.RPN.SA_CONFIG,
    #                                         IN_DIM=init_cfg.RPN.SA_CONFIG.MLPS[3][0][-1] + init_cfg.RPN.SA_CONFIG.MLPS[3][1][
    #                                             -1])
    #     self.context_fusion_3 = Fusion_Conv(
    #         init_cfg.RPN.SA_CONFIG.ATTN[2] + init_cfg.RPN.SA_CONFIG.MLPS[2][0][-1] + init_cfg.RPN.SA_CONFIG.MLPS[2][1][-1],
    #         init_cfg.RPN.SA_CONFIG.MLPS[2][0][-1] + init_cfg.RPN.SA_CONFIG.MLPS[2][1][-1])
    #     self.context_fusion_4 = Fusion_Conv(
    #         init_cfg.RPN.SA_CONFIG.ATTN[3] + init_cfg.RPN.SA_CONFIG.MLPS[3][0][-1] + init_cfg.RPN.SA_CONFIG.MLPS[3][1][-1],
    #         init_cfg.RPN.SA_CONFIG.MLPS[3][0][-1] + init_cfg.RPN.SA_CONFIG.MLPS[3][1][-1])


    def forward(self,i, img, l_xy_cor, li_xyz, li_features, li_index, l_features,xy ):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.

        Returns:
            torch.Tensor: Fused features of each point.
        """
        li_index = li_index.long().unsqueeze(-1).repeat(1, 1, 2)
        li_xy_cor = torch.gather(l_xy_cor[i], 1, li_index)

        image = self.Img_Block[i](img[i])
        if self.cross_conv:
            image= self.cross_fusion(i, image, li_xyz, li_features, li_xy_cor)
        img_gather_feature = Feature_Gather(image, li_xy_cor)  # , scale= 2**(i+1))
        li_features = self.Fusion_Conv[i](li_features, img_gather_feature)
        l_xy_cor.append(li_xy_cor)
        img.append(image)
        return img, li_xyz, li_features, l_xy_cor



    def cross_fusion(self, i, image, li_features, li_xy_cor):
        if self.P2I_path:
            first_img_gather_feature = Feature_Gather(image, li_xy_cor)  # , scale= 2**(i+1))
            image = self.cross_conv[i](li_features, first_img_gather_feature, li_xy_cor, image)
        else:
            img_shape = image.shape
            project_point2img_feature = grid_sample_reverse(li_features, li_xy_cor, img_shape)
            image = self.cross_conv[i](project_point2img_feature, image)
        return image



        #
        # if cfg.USE_SELF_ATTENTION:
        #     if i == 2:
        #         # Get context visa self-attention
        #         l_context_3 = self.context_conv3(batch_size, li_features, li_xyz)
        #         # Concatenate
        #         # li_features = torch.cat([li_features, l_context_3], dim=1)
        #         li_features = self.context_fusion_3(li_features, l_context_3)
        #     if i == 3:
        #         # Get context via self-attention
        #         l_context_4 = self.context_conv4(batch_size, li_features, li_xyz)
        #         # Concatenate
        #         # li_features = torch.cat([li_features, l_context_4], dim=1)
        #         li_features = self.context_fusion_4(li_features, l_context_4)


    def de_fusion(self,img,l_features,xy):
        deconv = []
        for i in range(len(self.img_channels) - 1):
            deconv.append(self.de_conv[i](img[i + 1]))
        de_concat = torch.cat(deconv, dim=1)

        img_fusion = ConvModule(de_concat)
        img_fusion_gather_feature = Feature_Gather(img_fusion, xy)
        l_features[0] = self.final_fusion_img_point(l_features[0], img_fusion_gather_feature)

        return  l_features[0], img_fusion

    # def obtain_mlvl_feats(self, img_feats, pts, img_metas):
    #     """Obtain multi-level features for each point.
    #
    #     Args:
    #         img_feats (list(torch.Tensor)): Multi-scale image features produced
    #             by image backbone in shape (N, C, H, W).
    #         pts (list[torch.Tensor]): Points of each sample.
    #         img_metas (list[dict]): Meta information for each sample.
    #
    #     Returns:
    #         torch.Tensor: Corresponding image features of each point.
    #     """
    #     if self.lateral_convs is not None:
    #         img_ins = [
    #             lateral_conv(img_feats[i])
    #             for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
    #         ]
    #     else:
    #         img_ins = img_feats
    #     img_feats_per_point = []
    #     # Sample multi-level features
    #     for i in range(len(img_metas)):
    #         mlvl_img_feats = []
    #         for level in range(len(self.img_levels)):
    #             mlvl_img_feats.append(
    #                 self.sample_single(img_ins[level][i:i + 1], pts[i][:, :3],
    #                                    img_metas[i]))
    #         mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1)
    #         img_feats_per_point.append(mlvl_img_feats)
    #
    #     img_pts = torch.cat(img_feats_per_point, dim=0)
    #     return img_pts
    #
    # def sample_single(self, img_feats, pts, img_meta):
    #     """Sample features from single level image feature map.
    #
    #     Args:
    #         img_feats (torch.Tensor): Image feature map in shape
    #             (1, C, H, W).
    #         pts (torch.Tensor): Points of a single sample.
    #         img_meta (dict): Meta information of the single sample.
    #
    #     Returns:
    #         torch.Tensor: Single level image features of each point.
    #     """
    #     # TODO: image transformation also extracted
    #     img_scale_factor = (
    #         pts.new_tensor(img_meta['scale_factor'][:2])
    #         if 'scale_factor' in img_meta.keys() else 1)
    #     img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
    #     img_crop_offset = (
    #         pts.new_tensor(img_meta['img_crop_offset'])
    #         if 'img_crop_offset' in img_meta.keys() else 0)
    #     proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
    #     img_pts = point_sample(
    #         img_meta=img_meta,
    #         img_features=img_feats,
    #         points=pts,
    #         proj_mat=pts.new_tensor(proj_mat),
    #         coord_type=self.coord_type,
    #         img_scale_factor=img_scale_factor,
    #         img_crop_offset=img_crop_offset,
    #         img_flip=img_flip,
    #         img_pad_shape=img_meta['input_shape'][:2],
    #         img_shape=img_meta['img_shape'][:2],
    #         aligned=self.aligned,
    #         padding_mode=self.padding_mode,
    #         align_corners=self.align_corners,
    #     )
    #     return img_pts
    #


class P2I_Layer(nn.Module):
    """Point clouds to Image path.

    """
    def __init__(self, channels):
        super(P2I_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.ic // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.pc, self.pc, 1),
                                   nn.BatchNorm1d(self.pc),
                                   nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, img_feats, point_feats):
        batch = img_feats.size(0)
        img_feats_f = img_feats.transpose(1, 2).contiguous().view(-1, self.ic)  # BCN->BNC->(BN)C
        point_feats_f = point_feats.transpose(1, 2).contiguous().view(-1, self.pc)  # BCN->BNC->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feats_f)
        rp = self.fc2(point_feats_f)
        att = F.sigmoid(self.fc3(F.tanh(ri + rp)))  # BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1)  # B1N
        # print(img_feas.size(), att.size())

        point_feats_new = self.conv1(point_feats)
        out = point_feats_new * att

        return out


class I2P_Layer(nn.Module):
    """Image to Point clouds path.

    """
    def __init__(self, channels):
        super(I2P_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),  #####
                                    nn.BatchNorm1d(self.pc),  ####
                                    nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)


    def forward(self, img_feas, point_feas):
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        att = F.sigmoid(self.fc3(F.tanh(ri + rp))) # BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1) # B1N
        # print(img_feas.size(), att.size())

        img_feas_new = self.conv1(img_feas)
        out = img_feas_new * att

        return out


class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Fusion_Conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        #print(point_features.shape, img_features.shape)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


class Fusion_Cross_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Fusion_Cross_Conv, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = nn.Conv2d(self.inplanes, self.outplanes, stride=1, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(outplanes)

    def forward(self, point_features, img_features):
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features

class Fusion_Cross_Conv_Gate(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Fusion_Cross_Conv_Gate, self).__init__()
        self.P2I_Layer = P2I_Layer(channels=[inplanes_I, inplanes_P])
        self.inplanes = inplanes_I + inplanes_P
        self.outplanes = outplanes
        self.conv1 = nn.Conv2d(self.inplanes, self.outplanes, stride=1, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.outplanes)

    def forward(self, point_features, img_features, li_xy_cor, image):

        point_features = self.P2I_Layer(img_features, point_features)

        point2img_feature = grid_sample_reverse(point_features, li_xy_cor, img_shape=image.shape)

        fusion_features = torch.cat([point2img_feature, image], dim=1)

        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features

class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Fusion_Conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):

        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features

class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()

        self.I2P_Layer = I2P_Layer(channels=[inplanes_I, inplanes_P])
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        img_features = self.I2P_Layer(img_features, point_features)

        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features

def Feature_Gather(feature_map, xy):
    """
    :param xy:(B,N,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """

    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)

    interpolate_feature = grid_sample(feature_map, xy)  # (B,C,1,N)

    return interpolate_feature.squeeze(2) # (B,C,N)


def grid_sample_reverse(point_feature, xy, img_shape):

    # print('#######point_feature:', point_feature.shape)
    # print('#######xy:', xy.shape)
    # print('#######size:', size)
    size = [i for i in img_shape]
    size[1] = point_feature.shape[1]
    point2img = grid_sample(input=point_feature, grid=xy, output= size, mode='bilinear', padding_mode='zeros', align_corners=None)

    return point2img