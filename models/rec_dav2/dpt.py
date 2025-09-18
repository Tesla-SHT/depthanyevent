import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet
from .submodules import RecurrentConvLayer

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class RecurrentDPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False,
        activation='relu',
        recurrent_block_type='convlstm'
    ):
        super(RecurrentDPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        self.activation = activation
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.gru_layer_1 = RecurrentConvLayer(features, features, recurrent_block_type=recurrent_block_type)
        self.gru_layer_2 = RecurrentConvLayer(features, features, recurrent_block_type=recurrent_block_type)
        self.gru_layer_3 = RecurrentConvLayer(features, features, recurrent_block_type=recurrent_block_type)
        self.gru_layer_4 = RecurrentConvLayer(features, features, recurrent_block_type=recurrent_block_type)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
        )

        if activation == 'relu':
            self.scratch.final_activation = nn.Sequential(
                nn.ReLU(True),
                nn.Identity()
            )
        elif activation == 'sigmoid':
            self.scratch.final_activation = nn.Sigmoid()
        elif activation == 'softplus':
            self.scratch.final_activation = nn.Softplus()
        else:
            raise ValueError(f"Activation {activation} not supported")
    
    def forward(self, out_features, patch_h, patch_w, prev_states=None):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        layer_4_rn_states, layer_3_rn_states, layer_2_rn_states, layer_1_rn_states = prev_states if prev_states is not None else (None, None, None, None)

        layer_4_rn, layer_4_rn_states = self.gru_layer_4(layer_4_rn, layer_4_rn_states)
        layer_3_rn, layer_3_rn_states = self.gru_layer_3(layer_3_rn, layer_3_rn_states)
        layer_2_rn, layer_2_rn_states = self.gru_layer_2(layer_2_rn, layer_2_rn_states)
        layer_1_rn, layer_1_rn_states = self.gru_layer_1(layer_1_rn, layer_1_rn_states)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        full_features = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=False)
        out = self.scratch.output_conv2(full_features)
        out = self.scratch.final_activation(out)
        
        return out, (layer_4_rn_states, layer_3_rn_states, layer_2_rn_states, layer_1_rn_states)


class RecurrentDepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        activation='relu',
        scale_factor=1.0,
        inv_prediction=False,
        input_size_width=518,
        input_size_height=518,
        freeze_encoder=False,
        input_channels=3,
        recurrent_block_type='convlstm'
    ):
        super(RecurrentDepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.scale_factor = scale_factor
        self.activation = activation
        self.inv_prediction = inv_prediction
        self.input_size_width = input_size_width
        self.input_size_height = input_size_height
        self.input_channels = input_channels

        if self.input_channels != 3:
            self.input_mapping = nn.Sequential(
                nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(16),
                nn.ReLU(True),
                nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(3),
                nn.ReLU(True),
            )
        else:
            self.input_mapping = nn.Identity()
        self.pretrained = DINOv2(model_name=encoder)
        self.depth_head = RecurrentDPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, activation=activation, recurrent_block_type=recurrent_block_type)

        if freeze_encoder:
            self.freeze_encoder()
            self.partial_freeze_head()
    
    def freeze_encoder(self):
        for param in self.pretrained.parameters():
            param.requires_grad = False

    def partial_freeze_head(self):
        for param in self.depth_head.parameters():
            param.requires_grad = False
        
        for _component in [self.depth_head.gru_layer_1, self.depth_head.gru_layer_2, self.depth_head.gru_layer_3, self.depth_head.gru_layer_4]:
            for param in _component.parameters():
                param.requires_grad = True

    def forward(self, x, prev_states=None):
        x = self.input_mapping(x)

        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth, new_states = self.depth_head(features, patch_h, patch_w, prev_states)
        
        if self.inv_prediction:
            if self.activation == 'relu':
                depth = 1.0 / (depth + 1e-6)
            elif self.activation == 'sigmoid':
                depth = 1.0 - depth

        depth = depth * self.scale_factor

        return depth, new_states
    
    def infer_image(self, raw_image, input_size_width=None, input_size_height=None, prev_states=None):
        if input_size_width is None:
            input_size_width = self.input_size_width
        if input_size_height is None:
            input_size_height = self.input_size_height

        image, (h, w), (final_h, final_w) = self.image2tensor(raw_image, input_size_width, input_size_height)
        
        depth, new_states = self.forward(image, prev_states)
        
        depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=False)

        return depth, new_states
    
    def image2tensor(self, raw_image, input_size_width=518, input_size_height=518):
        #B C H W
        h, w = raw_image.shape[-2], raw_image.shape[-1]

        resize = Resize(
                width=input_size_width,
                height=input_size_height,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            )
        
        final_w, final_h = resize.get_size(w,h)

        if not isinstance(final_w, int):
            final_w = int(final_w)
        if not isinstance(final_h, int):
            final_h = int(final_h)

        #print(f"W: {final_w}; H: {final_h}")

        image = F.interpolate(raw_image, (final_h, final_w), mode='bicubic', align_corners=False)
        
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        for i, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
            image[:,i].sub_(m).div_(s)

        return image, (h, w), (final_h, final_w)
