import torch.nn as nn
import torch.nn.functional as F
import torch
from .compact_bilinear_pool import CompactBilinearPooling


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0, dis_type = 'cos'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.dis_type = dis_type

    def forward(self, output1, output2, label):
        '''
        :param output1: N x feature_dim
        :param output2: N x feature_dim
        :param label: N
        :return:
        '''

        label = label.float()
        if self.dis_type == 'pairwise':
            distance = F.pairwise_distance(output1, output2)
            loss_contrastive = torch.mean((label) * torch.pow(distance, 2) +
                                          (1-label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        else:
            distance = 1.0 - F.cosine_similarity(output1, output2)  # self.margin in [0.0,2.0]
            loss_contrastive = torch.mean((label) * distance +
                                          (1 - label) * torch.clamp(self.margin - distance, min=0.0))

        return loss_contrastive


class System(nn.Module):
    """
    A,B,C Systems in
    "NOISE-TOLERANT AUDIO-VISUAL ONLINE PERSON VERIFICATION USING AN ATTENTION-BASED NEURAL NETWORK FUSION"
    The D system is the concat version of system C
    System E is from: "GATED MULTIMODAL UNITS FOR INFORMATION FUSION"
    System F is from: "Compact Bilinear Pooling"
    """

    def __init__(self, system_type='C', emb_dim_in=512, emb_dim_out=512, norm=True, layer=1,
                 attention_layer=1, trans_layer=0, mid_att_dim=64, cbp_dim=512, dropout_p=0.0,
                 system_A_layer=None, do_tanh=True):
        super(System, self).__init__()
        self.system_type = system_type
        self.norm = norm  # L2 normalize of input feature
        self.do_tanh = do_tanh  # for system E

        if system_type == 'A':
            self.linears_A = nn.Sequential()
            if system_A_layer is None:
                layer_num = 0
            else:
                system_A_layer.insert(0, emb_dim_in * 2)
                layer_num = len(system_A_layer)

            if layer_num == 2:
                self.linears_A.add_module(str(0), nn.Linear(system_A_layer[0], system_A_layer[1]))
            elif layer_num > 2:
                for i in range(0, layer_num - 2):
                    self.linears_A.add_module(str(i), Forward_Block(system_A_layer[i], system_A_layer[i+1]))
                self.linears_A.add_module(str(layer_num-2), nn.Linear(system_A_layer[-2], system_A_layer[-1]))

        elif system_type == 'B':
            self.linear_face = nn.Sequential()
            self.linear_utt = nn.Sequential()
            for i in range(layer - 1):
                self.linear_face.add_module('layer_' + str(i), Forward_Block(emb_dim_in,emb_dim_in, dropout_p))
                self.linear_utt.add_module('layer_' + str(i), Forward_Block(emb_dim_in, emb_dim_in, dropout_p))

            self.linear_face.add_module('layer_' + str(layer - 1), nn.Linear(emb_dim_in, emb_dim_out))
            self.linear_utt.add_module('layer_' + str(layer - 1), nn.Linear(emb_dim_in, emb_dim_out))

        else:
            # Embedding transformation before attention
            self.linear_face = nn.Sequential()
            self.linear_utt = nn.Sequential()
            if layer > 0:
                for i in range(layer - 1):
                    self.linear_face.add_module('layer_' + str(i), Forward_Block(emb_dim_in, emb_dim_out, dropout_p))
                    self.linear_utt.add_module('layer_' + str(i), Forward_Block(emb_dim_in, emb_dim_out, dropout_p))

                if layer == 1:
                    self.linear_face.add_module('layer_' + str(layer - 1), nn.Linear(emb_dim_in, emb_dim_out))
                    self.linear_utt.add_module('layer_' + str(layer - 1), nn.Linear(emb_dim_in, emb_dim_out))
                elif layer > 1:
                    self.linear_face.add_module('layer_' + str(layer - 1), nn.Linear(emb_dim_out, emb_dim_out))
                    self.linear_utt.add_module('layer_' + str(layer - 1), nn.Linear(emb_dim_out, emb_dim_out))

        if system_type == 'C' or system_type == 'D':
            if attention_layer == 1:
                self.attention_W = nn.Linear(emb_dim_in * 2, 2)
            else:
                dim_list = []
                for i in range(attention_layer):
                    dim_list.append(emb_dim_in * 2 // (2**i))
                self.attention_W = nn.Sequential()
                for i in range(attention_layer-1):
                    self.attention_W.add_module(str(i), Forward_Block(dim_list[i], dim_list[i+1]))
                self.attention_W.add_module(str(attention_layer-1), nn.Linear(dim_list[-1], 2))

        if system_type == 'E':
            # We assert the attention layer is 2
            self.attention_W = nn.Sequential(
                Forward_Block(emb_dim_in * 2, mid_att_dim),
                nn.Linear(mid_att_dim, emb_dim_out)
            )

        if system_type == 'F':
            self.compact_bilinear_pooling = CompactBilinearPooling(emb_dim_out, emb_dim_out, cbp_dim)

        # Final transform layer
        self.final_transform = nn.Sequential()
        if system_type == 'D': emb_dim_out *= 2
        dim_list = []
        for i in range(trans_layer + 1):
            dim_list.append(emb_dim_out)
        if system_type == 'F': dim_list[0] = cbp_dim
        if trans_layer > 0:
            for i in range(trans_layer - 1):
                self.final_transform.add_module('layer_' + str(i), Forward_Block(dim_list[i], dim_list[i+1], dropout_p))
            if trans_layer >= 1:
                self.final_transform.add_module('layer_' + str(trans_layer - 1), nn.Linear(dim_list[-2], dim_list[-1]))

    def forward(self, face_input, utt_input):
        if self.norm:
            face_input = F.normalize(face_input, p=2, dim=1)
            utt_input = F.normalize(utt_input, p=2, dim=1)

        if self.system_type == 'A':
            concat = torch.cat((face_input, utt_input), dim=1)
            out = self.linears_A(concat)
        elif self.system_type == 'B':
            face_trans = self.linear_face(face_input)
            utt_trans = self.linear_utt(utt_input)
            out = (face_trans + utt_trans) / 2
        elif self.system_type == 'C':
            concat = torch.cat((face_input, utt_input), dim=1)
            attention = self.attention_W(concat)  # N x 2
            attention = F.softmax(attention, dim=1) # N x 2
            face_trans = self.linear_face(face_input)
            utt_trans = self.linear_utt(utt_input)
            out = face_trans * attention[:,0].reshape(-1,1) + utt_trans * attention[:,1].reshape(-1,1)
            out = self.final_transform(out)
        elif self.system_type == 'D':
            concat = torch.cat((face_input, utt_input), dim=1)
            attention = self.attention_W(concat)  # N x 2
            attention = F.softmax(attention, dim=1)  # N x 2
            face_trans = self.linear_face(face_input) * attention[:,0].reshape(-1,1)
            utt_trans = self.linear_utt(utt_input) * attention[:,1].reshape(-1,1)
            out = torch.cat((face_trans, utt_trans), dim=1)
            out = self.final_transform(out)
        elif self.system_type == 'E':
            concat = torch.cat((face_input, utt_input), dim=1)
            attention = torch.sigmoid(self.attention_W(concat))  # N x emb_dim
            if self.do_tanh:
                face_trans = torch.tanh(self.linear_face(face_input))  # N x emb_dim
                utt_trans = torch.tanh(self.linear_utt(utt_input))  # N x emb_dim
            else:
                face_trans = self.linear_face(face_input)  # N x emb_dim
                utt_trans = self.linear_utt(utt_input)  # N x emb_dim
            out = face_trans * attention + (1.0 - attention) * utt_trans
            out = self.final_transform(out)
        else:
            face_trans = self.linear_face(face_input)  # N x emb_dim
            utt_trans = self.linear_utt(utt_input)  # N x emb_dim
            cbp_out = self.compact_bilinear_pooling(face_trans, utt_trans)
            out = self.final_transform(cbp_out)

        return out, face_trans, utt_trans

    def get_attention(self, face_input, utt_input):
        if self.norm:
            face_input = F.normalize(face_input, p=2, dim=1)
            utt_input = F.normalize(utt_input, p=2, dim=1)

        concat = torch.cat((face_input, utt_input), dim=1)
        attention = self.attention_W(concat)  # N x 2
        attention = F.softmax(attention, dim=1)  # N x 2
        return attention

    def get_face_emb(self, face_input):
        if self.norm:
            face_input = F.normalize(face_input, p=2, dim=1)

        face_trans = self.linear_face(face_input)
        return face_trans

    def get_utt_emb(self, utt_input):
        if self.norm:
            utt_input = F.normalize(utt_input, p=2, dim=1)

        utt_trans = self.linear_utt(utt_input)
        return utt_trans


class Forward_Block(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, p_val=0.0):
        super(Forward_Block, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=p_val)
        )

    def forward(self, x):
        return self.block(x)


class Fusion_Channel(nn.Module):
    """
    Implementation of fusion strategy in paper "Attention Fusion for Audio-Visual Person Verification Using Multi-Scale Features"
    """

    def __init__(self, face_channel_dims=[64, 128, 256], utt_channel_dims=[32, 64, 128], 
                        face_emb_in=512, utt_emb_in=512, emb_out=512):
        super(Fusion_Channel, self).__init__()
        
        assert len(face_channel_dims) == len(utt_channel_dims)

        self.linear_face = nn.Linear(face_emb_in, emb_out)
        self.linear_utt = nn.Linear(utt_emb_in, emb_out)

        self.attention_W = nn.ModuleList()
        for face_dim, utt_dim in zip(face_channel_dims, utt_channel_dims):
            self.attention_W.append(nn.Linear(face_dim+utt_dim, 2))

    def forward(self, face_feature, utt_feature, face_input, utt_input):
        '''
        face_feature and utt_feature input are tuples
        face_feature (N x feat_dim1, N x feat_dim2....)
        face_input N x emb_dim
        '''
        assert len(face_feature) == len(utt_feature)

        att_avg = face_input.new_zeros((face_input.shape[0], 2))
        for face_feat, utt_feat, att_W in zip(face_feature, utt_feature, self.attention_W):
            concat = torch.cat((face_feat, utt_feat), dim=1)
            attention = att_W(concat)  # N x 2
            att_avg =  att_avg + F.softmax(attention, dim=1) # N x 2
        att_avg = att_avg / len(face_feature)

        face_trans = self.linear_face(face_input)
        utt_trans = self.linear_utt(utt_input)
        out = face_trans * att_avg[:,0].reshape(-1,1) + utt_trans * att_avg[:,1].reshape(-1,1)

        return out



if __name__ == '__main__':
    # model = System(system_type = 'C', emb_dim_in = 512, emb_dim_out = 512, norm = True, layer = 2,
    #                attention_layer=3)
    # model.eval()

    # input1 = torch.randn(1, 512)
    # input2 = torch.randn(1, 512)

    # out = model(input1, input2)
    # att = model.get_attention(input1, input2)
    # print(att)
    # print(att[0][0].item())
    # print(att[0][1].item())


    model = Fusion_Channel()
    
    face_input = torch.randn(4, 512)
    utt_input = torch.randn(4, 512)

    face_feature  = (torch.randn(4, 64), torch.randn(4, 128), torch.randn(4, 256))
    utt_feature  = (torch.randn(4, 32), torch.randn(4, 64), torch.randn(4, 128))

    out = model(face_feature, utt_feature, face_input, utt_input)
    print(out.shape)


    