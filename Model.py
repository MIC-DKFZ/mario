# from pathlib import Path
from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_from_pretrained
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


# ***** RETFound *****

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class SiameseRetinaFound(nn.Module):
    def __init__(self, num_classes=4):
        super(SiameseRetinaFound, self).__init__()

        model = vit_large_patch16(num_classes=3, drop_path_rate=0, global_pool=True, )

        # Load the pretrained weights
        checkpoint = torch.load(
            "/home/a332l/E132-Projekte/Projects/2024_MICCAI_Mario_Challenge/pretraining/RETFound_oct_weights.pth",
            map_location='cpu')
        print("Load pre-trained checkpoint")
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)

        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

        self.encoder = model

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),  # Input size should be 1024 after feature subtraction
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x1, x2):
        x1 = self.encoder.patch_embed(x1)
        x2 = self.encoder.patch_embed(x2)

        combined_output = x1 - x2  # Feature subtraction

        x = self.classifier(combined_output)
        return x


# ***** BiomedCLIP *****

class SiameseBiomed(nn.Module):
    def __init__(self, num_classes=4):
        super(SiameseBiomed, self).__init__()

        self.encoder, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),  # Input size should be 1024 after concatenation
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x1, x2):
        x1 = self.encoder.encode_image(x1)
        x2 = self.encoder.encode_image(x2)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        combined_output = x1 - x2  # Feature subtraction

        x = self.classifier(combined_output)
        return x


# ***** ResNet *****    

class SiameseNetwork(nn.Module):
    def __init__(self, num_classes=4, encoder_weights_path=None, res50=False, mAE_pretrained=False,
                 contrastive_pretrained=False):
        super(SiameseNetwork, self).__init__()

        if encoder_weights_path:
            if mAE_pretrained:
                pretrained_model = mAE()
            elif contrastive_pretrained:
                pretrained_model = ResNetSimCLR(base_model="resnet18", out_dim=256)
                pretrained_model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                         bias=False)
            elif res50:
                pretrained_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                pretrained_model = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)

            pretrained_model.load_state_dict(torch.load(encoder_weights_path))

            if contrastive_pretrained:
                self.encoder = pretrained_model.features
            else:
                self.encoder = pretrained_model.encoder

        else:
            if res50:
                self.encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                self.encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        if res50:
            self.classifier = nn.Sequential(
                nn.Linear(2048 * 2, 1024),  # Input size should be 4096 after concatenation
                nn.ReLU(),
                nn.Linear(1024, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 2, 512),  # Input size should be 1024 after concatenation
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
            )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2):

        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        if isinstance(x1, tuple):
            x1 = x1[0]
            x2 = x2[0]
            x1 = self.global_avg_pool(x1)
            x2 = self.global_avg_pool(x2)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        combined_output = torch.cat((x1, x2), 1)  # The combined size should be 4096 for ResNet50 and 1024 for ResNet18

        x = self.classifier(combined_output)
        return x


class SiameseNetworkSub(nn.Module):
    def __init__(self, num_classes=4, encoder_weights_path=None, res50=False, mAE_pretrained=False,
                 contrastive_pretrained=False):
        super(SiameseNetworkSub, self).__init__()

        if encoder_weights_path:
            if mAE_pretrained:
                pretrained_model = mAE()
            elif contrastive_pretrained:
                pretrained_model = ResNetSimCLR(base_model="resnet50", out_dim=256)
                pretrained_model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                         bias=False)
            elif res50:
                pretrained_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                pretrained_model = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)

            pretrained_model.load_state_dict(torch.load(encoder_weights_path))

            if contrastive_pretrained:
                self.encoder = pretrained_model.features
            else:
                self.encoder = pretrained_model.encoder
        else:
            if res50:
                self.encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                self.encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        if res50 or mAE_pretrained or contrastive_pretrained:
            self.classifier = nn.Sequential(
                nn.Linear(2048, 1024),  # Corrected comment for feature size
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),  # Corrected comment for feature size
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
            )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        if isinstance(x1, tuple):
            x1 = x1[0]
            x2 = x2[0]

        x1 = self.global_avg_pool(x1)
        x2 = self.global_avg_pool(x2)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        combined_output = x1 - x2  # Feature subtraction

        x = self.classifier(combined_output)
        return x


# ******************
# ***** Task 2 *****
# ******************

class Task2SiameseNetwork(nn.Module):
    def __init__(self, num_classes=3, encoder_weights_path=None, res50=False, mAE_pretrained=False,
                 contrastive_pretrained=False, sub=False):
        super(Task2SiameseNetwork, self).__init__()

        if encoder_weights_path:
            if sub:
                pretrained_model = SiameseNetworkSub(num_classes=4, res50=res50, mAE_pretrained=mAE_pretrained,
                                                     contrastive_pretrained=contrastive_pretrained)
                pretrained_model.load_state_dict(torch.load(encoder_weights_path))
                self.encoder = pretrained_model.encoder
            else:
                pretrained_model = SiameseNetwork(num_classes=4, res50=res50, mAE_pretrained=mAE_pretrained,
                                                  contrastive_pretrained=contrastive_pretrained)
                pretrained_model.load_state_dict(torch.load(encoder_weights_path))
                self.encoder = pretrained_model.encoder
        else:
            self.encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        if res50 or mAE_pretrained or contrastive_pretrained:
            self.classifier = nn.Sequential(
                nn.Linear(2048, 1024),  # Corrected comment for feature size
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),  # Corrected comment for feature size
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
            )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        if isinstance(x1, tuple):
            x1 = x1[0]
            x2 = x2[0]

        x1 = self.global_avg_pool(x1)
        x2 = self.global_avg_pool(x2)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        combined_output = x1 - x2  # Feature subtraction

        x = self.classifier(combined_output)
        return x


class Task2Biomed(nn.Module):
    def __init__(self, num_classes=3, encoder_weights_path=None, res50=False, biomed=False):
        super(Task2Biomed, self).__init__()

        pretrained_model = SiameseBiomed(num_classes=4)
        pretrained_model.load_state_dict(torch.load(encoder_weights_path))
        self.encoder = pretrained_model.encoder

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),  # Input size should be 4096 after concatenation
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.encoder.encode_image(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x


class Task2RetinaFound(nn.Module):
    def __init__(self, num_classes=3, encoder_weights_path=None):
        super(Task2RetinaFound, self).__init__()

        model = vit_large_patch16(num_classes=3, drop_path_rate=0, global_pool=False, )

        # Load the pretrained weights
        checkpoint = torch.load(
            "/home/a332l/E132-Projekte/Projects/2024_MICCAI_Mario_Challenge/pretraining/RETFound_oct_weights.pth",
            map_location='cpu')
        print("Load pre-trained checkpoint")
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        model.load_state_dict(checkpoint_model, strict=False)

        # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

        self.encoder = model

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),  # Input size should be 4096 after concatenation
        )

    def forward(self, x):

        x = self.encoder.forward_features(x)
        x = self.classifier(x)

        return x


# ***********************************
# ***** CONTRASTIVE PRETRAINING *****
# ***********************************

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d),
                            "resnet50": models.resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x


# ***********************
# ***** Autoencoder *****
# ***********************

def create_mask(image_size, patch_size, mask_ratio):
    """Creates a mask with randomly placed patches.

    Args:
        image_size: Tuple of image dimensions (height, width).
        patch_size: Size of the patches to be masked.
        mask_ratio: Desired ratio of masked pixels.

    Returns:
        A torch tensor representing the mask.
    """

    h, w = image_size
    total_pixels = h * w
    num_masked_pixels = int(mask_ratio * total_pixels)
    num_patches = int(num_masked_pixels / (patch_size * patch_size))

    mask = torch.zeros(h, w)
    for _ in range(num_patches):
        x = torch.randint(0, h - patch_size + 1, (1,)).item()
        y = torch.randint(0, w - patch_size + 1, (1,)).item()
        mask[x:x + patch_size, y:y + patch_size] = 1

    return mask


class PretrainedResNetEncoder(nn.Module):
    def __init__(self, mask_ratio=0.75, use_mask=True):
        super(PretrainedResNetEncoder, self).__init__()
        # resnet = models.resnet18(pretrained=True)
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.initial = nn.Sequential(*list(resnet.children())[:3])
        self.mp = nn.Sequential(*list(resnet.children())[3:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.mask_ratio = mask_ratio
        self.use_mask = use_mask

    def forward(self, x):
        if self.use_mask:
            mask = create_mask(x.shape[2:], 16, self.mask_ratio).to(x.device)
            mask = mask.unsqueeze(0).unsqueeze(0).expand_as(x)
            x = x * (1 - mask)
        else:
            mask = torch.zeros_like(x)

        skips = []
        x = self.initial(x)
        skips.append(x)
        x = self.mp(x)
        x = self.layer1(x)
        skips.append(x)
        x = self.layer2(x)
        skips.append(x)
        x = self.layer3(x)
        skips.append(x)
        x = self.layer4(x)
        skips.append(x)
        return x, skips, mask


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        # self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.layer1 = ResidualBlock(512, 256)
        # self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.layer2 = ResidualBlock(256, 128)
        # self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.layer3 = ResidualBlock(128, 64)
        # self.upconv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # self.layer4 = ResidualBlock(128, 64)
        # self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.layer1 = ResidualBlock(2048, 1024)
        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.layer2 = ResidualBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.layer3 = ResidualBlock(512, 256)
        self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.layer4 = ResidualBlock(128, 64)
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skips):
        x = self.upconv1(x)
        x = torch.cat((x, skips[-2]), dim=1)
        x = self.layer1(x)

        x = self.upconv2(x)
        x = torch.cat((x, skips[-3]), dim=1)
        x = self.layer2(x)

        x = self.upconv3(x)
        x = torch.cat((x, skips[-4]), dim=1)
        x = self.layer3(x)

        x = self.upconv4(x)
        x = torch.cat((x, skips[-5]), dim=1)
        x = self.layer4(x)
        x = self.final_conv(x)
        return x


class mAE(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, mask_ratio=0.75, use_mask=False):
        super(mAE, self).__init__()
        self.encoder = PretrainedResNetEncoder(mask_ratio, use_mask)
        self.decoder = Decoder(out_channels)

    def forward(self, x):
        encoded, skips, mask = self.encoder(x)
        decoded = self.decoder(encoded, skips)
        return decoded, mask

    def loss(self, original, reconstruction, mask):
        return F.mse_loss(reconstruction * mask, original * mask)


# ***************************
# ***** OCT pretraining *****
# ***************************


class OCT_model(nn.Module):
    def __init__(self, n_classes=4):
        super(OCT_model, self).__init__()
        self.encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, input):
        x = self.encoder(input)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ********************
# ***** For UNet *****
# ********************


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, patch_size, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.patch_size = patch_size
        self.down_a = (DoubleConv(n_channels, 64))
        self.down_b = (DoubleConv(n_channels, 64))
        self.combine = (Up(128, 64))
        self.inc = (DoubleConv(64, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, 1))
        self.avgpool = nn.MaxPool2d((3, 3))
        if self.patch_size == 224:
            in_feats = 5476
        elif self.patch_size == 512:
            in_feats = 28900
        elif self.patch_size == 1024:
            in_feats = 116281
        self.fc = nn.Linear(in_feats, self.n_classes)

    def forward(self, x):
        a = x[:, :6, :, :]
        a = a.squeeze()
        b = x[:, 6:, :, :]
        b = b.squeeze()
        a_out = self.down_a(a)
        b_out = self.down_b(b)
        x_new = self.combine(a_out, b_out)
        x1 = self.inc(x_new)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.avgpool(logits)
        logits = torch.flatten(logits, 1)
        logits = self.fc(logits)
        return logits


class UNet_all_channels(nn.Module):
    def __init__(self, n_channels, n_classes, patch_size, bilinear=False):
        super(UNet_all_channels, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.patch_size = patch_size
        self.down_a = (DoubleConv(n_channels, 64))
        self.inc = (DoubleConv(64, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, 1))
        self.avgpool = nn.MaxPool2d((3, 3))
        if self.patch_size == 224:
            in_feats = 5476
        elif self.patch_size == 512:
            in_feats = 28900
        elif self.patch_size == 1024:
            in_feats = 116281
        self.fc = nn.Linear(in_feats, self.n_classes)

    def forward(self, x):
        x = x.squeeze()
        x = self.down_a(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.avgpool(logits)
        logits = torch.flatten(logits, 1)
        logits = self.fc(logits)
        return logits


def get_model(config):
    if config.sub:
        return SiameseNetworkSub(num_classes=4, encoder_weights_path=config.encoder_weights_path, res50=config.res50,
                                 mAE_pretrained=config.mAE_pretrained,
                                 contrastive_pretrained=config.contrastive_pretrained)
    elif config.biomed:
        return SiameseBiomed(num_classes=4)
    elif config.RETFound:
        return SiameseRetinaFound(num_classes=4)
    else:
        return SiameseNetwork(num_classes=4, encoder_weights_path=config.encoder_weights_path, res50=config.res50,
                              mAE_pretrained=config.mAE_pretrained,
                              contrastive_pretrained=config.contrastive_pretrained)
