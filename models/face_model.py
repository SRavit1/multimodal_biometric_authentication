from .projections import ArcMarginProduct
import torch.nn as nn
import torchvision
import torch

class FaceReg(nn.Module):

    def __init__(self, embedding_dim = 256, class_num = 8631,extractor_name = 'resnet34',
                 scale=32, margin=0.1, pretrained = False, r = 16, angular = True):
        super().__init__()

        self.angular = angular

        if 'se' in extractor_name:
            extractor_name = extractor_name[2:]  # seresnet --> resnet
            resnet = eval('torchvision.models.{}(pretrained={})'.format(extractor_name, str(pretrained)))
            resnet_modules = list(resnet.children())
            self.resnet_base = SEResnet(resnet_modules[:-2], r)
        else:
            resnet = eval('torchvision.models.{}(pretrained={})'.format(extractor_name, str(pretrained)))
            resnet_modules = list(resnet.children())
            self.resnet_base = nn.Sequential(*resnet_modules[:-1])


        self.linear = nn.Linear(resnet_modules[-1].in_features,embedding_dim,bias=True)
        self.bn = nn.BatchNorm1d(embedding_dim, affine=False)

        if self.angular:
            self.projection = ArcMarginProduct(embedding_dim, class_num, scale, margin)
        else:
            self.projection = nn.Linear(embedding_dim, class_num)

    def forward(self, x, label):
        x = self.resnet_base(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.bn(x)

        if self.angular:
            x = self.projection(x, label)
        else:
            x = self.projection(x)

        return x

    def get_emb(self, x):
        x = self.resnet_base(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

    def get_emb_out(self, x, label):
        x = self.resnet_base(x)
        x = torch.flatten(x, 1)
        emb = self.linear(x)
        out = self.bn(emb)

        if self.angular:
            out = self.projection(out, label)
        else:
            out = self.projection(out)

        return emb, out

class SEBasicBlock(nn.Module):
    '''
    Basic Block for SEResnet
    Add Squeeze-and-Excitation part into residual block

    [1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu

    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507
    '''
    def __init__(self, resblock, r = 16):
        super().__init__()

        self.res = nn.Sequential()
        self.downsample = None
        self.relu = nn.ReLU(inplace=True)


        self.res_outchannels = 0 # the out channel dimension of resblock
        resmodules = list(resblock.children())


        count_conv = 0
        count_batchnorm = 0
        for child in resmodules:
            module_name = child.__class__.__name__

            if 'Conv' in module_name:
                count_conv += 1

                if self.res.__len__() != 0: self.res.add_module('relu' + str(count_conv-1), nn.ReLU(inplace=True))
                self.res.add_module('conv' + str(count_conv), child)
                self.res_outchannels = child.out_channels

            if 'BatchNorm' in module_name:
                count_batchnorm += 1
                self.res.add_module('bn' + str(count_batchnorm) , child)

            if 'Sequential' in module_name:
                self.downsample = child

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(self.res_outchannels, self.res_outchannels // r),
            nn.ReLU(inplace=True),
            nn.Linear(self.res_outchannels // r, self.res_outchannels),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        residual = self.res(x)

        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)


        if self.downsample is not None:
            identity = self.downsample(x)

        out = residual * excitation.expand_as(residual) + identity

        out = self.relu(out)
        return out


class SEResnet(nn.Module):

    def __init__(self, resnet_modules , r = 16):
        super().__init__()

        self.conv1 = nn.Sequential(*resnet_modules[:4])

        self.conv2_x = self.generate_seres_layer(resnet_modules[4], r)
        self.conv3_x = self.generate_seres_layer(resnet_modules[5], r)
        self.conv4_x = self.generate_seres_layer(resnet_modules[6], r)
        self.conv5_x = self.generate_seres_layer(resnet_modules[7], r)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avgpool(x)

        return x


    def generate_seres_layer(self, reslayer, r):
        return_layer = nn.Sequential()

        for i,block in enumerate(list(reslayer.children())):
            return_layer.add_module('block'+str(i+1), SEBasicBlock(block,r))

        return return_layer


if __name__ == '__main__':
    # import torch
    # input = torch.randn(1, 3, 112, 96)
    # label = torch.randint(0,123,(1,))
    # model = FaceReg(embedding_dim = 256, class_num = 8631,extractor_name = 'seresnet101',
    #              scale=32, margin=0.1, pretrained = True, r = 16)
    # for name,param in model.resnet_base.named_parameters():
    #     if 'excitation'  in name:
    #         print(name)

    model = FaceReg(embedding_dim=256, class_num=8631, extractor_name='resnet34',
                    scale=32, margin=0.1, pretrained=False, r=16)
    print(model)


