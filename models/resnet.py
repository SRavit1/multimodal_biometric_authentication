'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, m_channels=32, feat_dim=40, embed_dim=128):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels*8, num_blocks[3], stride=2)
        self.embedding = nn.Linear(int(feat_dim/8) * m_channels * 16 * block.expansion, embed_dim)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        pooling_mean = torch.mean(out, dim=-1)
        pooling_std = torch.sqrt(torch.var(out, dim=-1) + 1e-10)
        out = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                         torch.flatten(pooling_std, start_dim=1)), 1)
        embedding = self.embedding(out)
        return embedding 

    def get_feature_out(self, x):
        '''
        Implementation of paper "Attention Fusion for Audio-Visual Person Verification Using Multi-Scale Features"
        '''
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        res1_out = self.layer1(out)
        res2_out = self.layer2(res1_out)
        res3_out = self.layer3(res2_out)
        out = self.layer4(res3_out)

        pooling_mean = torch.mean(out, dim=-1)
        pooling_std = torch.sqrt(torch.var(out, dim=-1) + 1e-10)
        out = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                         torch.flatten(pooling_std, start_dim=1)), 1)
        embedding = self.embedding(out)

        res1_feature = torch.flatten(F.adaptive_avg_pool2d(res1_out, (1,1)) , start_dim=1)
        res2_feature = torch.flatten(F.adaptive_avg_pool2d(res2_out, (1,1)) , start_dim=1)
        res3_feature = torch.flatten(F.adaptive_avg_pool2d(res3_out, (1,1)) , start_dim=1)
        return embedding, res1_feature, res2_feature, res3_feature


class ResNet_Joint_Base(nn.Module):
    # Resnet architecture for audio-visual input fusion baseline
    def __init__(self, block, num_blocks, m_channels=64, feat_dim=40, embed_dim=128):
        super(ResNet_Joint_Base, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, m_channels, kernel_size=(1,7), stride=(1,2), padding=(0,3),
                      bias=False),
            nn.BatchNorm2d(m_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(m_channels, m_channels, kernel_size=(1,7), stride=(1,2), padding=(0,3),
                      bias=False),
            nn.BatchNorm2d(m_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(m_channels, m_channels, kernel_size=(1,7), stride=(1,2), padding=(0,3),
                      bias=False),
            nn.BatchNorm2d(m_channels),
            nn.ReLU(inplace=True))
        
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels*8, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(m_channels * 8 * block.expansion, embed_dim)

        self.fc = nn.Linear(m_channels * 8 * 3, embed_dim)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze_(1)
        output = self.conv1(x)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        pooling_mean = torch.mean(output, dim=-1)
        output = torch.flatten(pooling_mean, start_dim=1)
        output = self.fc(output)
        return output
        
        # output = self.avg_pool(output)
        # output = output.view(output.size(0), -1)
        # output = self.fc(output)

        # return output

class ResNet_ATT(nn.Module):
    def __init__(self, block, num_blocks, m_channels=32, feat_dim=40, embed_dim=128):
        super(ResNet_ATT, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dense_dim = int(feat_dim/8) * m_channels * 8 * block.expansion 

        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels*8, num_blocks[3], stride=2)

        self.mha = nn.MultiheadAttention(self.dense_dim, 5)

        self.embedding = nn.Linear(self.dense_dim * 2, embed_dim)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze_(1)
        print (x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        print (out.size())
        out = out.transpose(1, 3)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(0, 1)

        out, _ = self.mha(out, out, out)
        out = out.transpose(0, 1)
        out = out.transpose(1, 2)

        pooling_mean = torch.mean(out, dim=-1)
        pooling_std = torch.sqrt(torch.var(out, dim=-1) + 1e-10)
        print (pooling_mean.size())
        print (torch.flatten(pooling_mean, start_dim=1).size())
        out = torch.cat((pooling_mean, pooling_std), 1)
        embedding = self.embedding(out)
        return embedding 

class ResNet_MP(nn.Module):
    def __init__(self, block, num_blocks, m_channels=32, feat_dim=40, embed_dim=128):
        super(ResNet_MP, self).__init__()
        """
        With multi-level poolling
        """
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels*8, num_blocks[3], stride=2)
        self.embedding1 = nn.Linear(int(feat_dim) * m_channels * 2 * block.expansion, int(embed_dim/8))
        self.embedding2 = nn.Linear(int(feat_dim/2) * m_channels * 4 * block.expansion, int(embed_dim/8))
        self.embedding3 = nn.Linear(int(feat_dim/4) * m_channels * 8 * block.expansion, int(embed_dim/4))
        self.embedding = nn.Linear(int(feat_dim/8) * m_channels * 16 * block.expansion, int(embed_dim/2))


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def pooling(self, out):
        pooling_mean = torch.mean(out, dim=-1)
        pooling_std = torch.sqrt(torch.var(out, dim=-1) + 1e-10)
        out = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                         torch.flatten(pooling_std, start_dim=1)), 1)
        return out

    def forward(self, x):
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out1 = self.pooling(out)
        out = self.layer2(out)
        out2 = self.pooling(out)
        out = self.layer3(out)
        out3 = self.pooling(out)
        out = self.layer4(out)

        out = self.pooling(out)

        embedding1 = self.embedding1(out1)
        embedding2 = self.embedding2(out2)
        embedding3 = self.embedding3(out3)
        embedding = self.embedding(out)

        return torch.cat((embedding, embedding1, embedding2, embedding3), 1)

def ResNet18(feat_dim, embed_dim):
    return ResNet(BasicBlock, [2,2,2,2], feat_dim=feat_dim, embed_dim=embed_dim)

def ResNet34(feat_dim, embed_dim, m_channels=32):
    return ResNet(BasicBlock, [3,4,6,3], feat_dim=feat_dim, embed_dim=embed_dim, m_channels=m_channels)

def ResNet34_Joint_Base(feat_dim, embed_dim, m_channels=32):
    return ResNet_Joint_Base(BasicBlock, [3,4,6,3], feat_dim=feat_dim, embed_dim=embed_dim, m_channels=m_channels)

def ResNet34_MP(feat_dim, embed_dim):
    return ResNet_MP(BasicBlock, [3,4,6,3], feat_dim=feat_dim, embed_dim=embed_dim)

def ResNet34_ATT(feat_dim, embed_dim):
    return ResNet_ATT(BasicBlock, [3,4,6,3], feat_dim=feat_dim, embed_dim=embed_dim)

def ResNet50(feat_dim, embed_dim, m_channels=32):
    return ResNet(Bottleneck, [3,4,6,3], feat_dim=feat_dim, embed_dim=embed_dim, m_channels=m_channels)

def ResNet101(feat_dim, embed_dim):
    return ResNet(Bottleneck, [3,4,23,3], feat_dim=feat_dim, embed_dim=embed_dim)

def ResNet152(feat_dim, embed_dim):
    return ResNet(Bottleneck, [3,8,36,3], feat_dim=feat_dim, embed_dim=embed_dim)


if __name__ == '__main__':
    net = ResNet34_Joint_Base(40, 256)
    x = torch.randn(16,40,400)
    out = net(x)



    
