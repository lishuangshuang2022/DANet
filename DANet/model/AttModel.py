from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import GCN
import utils.util as util
import numpy as np
import torch.nn.functional as F

class Mlp_Trans(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob=0.5, out_act=True):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.dropout = nn.Dropout(p=do_prob)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.init_weights()
        self.out_act = out_act

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x_skip = x.transpose(1, 2)  # torch.Size([32, 31, 256])
        x = self.fc1(x_skip)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x) + x_skip
        return x.transpose(1, 2)


class AttModel(Module):

    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10):
        super(AttModel, self).__init__()

        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        assert kernel_size == 10

        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model//16, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model//16, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU(),
                                   Mlp_Trans(d_model,d_model,d_model,do_prob=0.2,out_act=False))

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model//16, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model//16, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU(),
                                   Mlp_Trans(d_model,d_model,d_model,do_prob=0.2,out_act=False))

        self.gcn = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

    def generate_square_subsequent_mask(self, sz: int):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf')
        """
        # mask = (torch.triu(torch.ones(sz, sz))).T
        mask_original = torch.eye(sz, sz)
        mask = (torch.triu(torch.ones(sz, sz),diagonal=1))
        for i in range(sz):#lie
            num=(1+i)*(i)/2
            for j in range(i):#hang
                if i > 1:
                    mask[j, i] = (j+1)*mask[j, i]/num
                else:
                    mask[j, i] = mask[j, i]
        # mask_numpylook = mask.cpu().data.numpy()
        # mask_future =torch.rot90(torch.rot90(mask))

        return mask.cuda(),mask_original.cuda()#.to(self.dct_n.device)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """

        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        dct_n = self.dct_n#20
        src = src[:, :input_n]  # [bs,in_n,dim]torch.Size([32, 50, 48])
        src_tmp = src.cuda().clone()
        bs = src.shape[0]#32
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()#torch.Size([32, 48, 40])
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()#torch.Size([32, 48, 10])

        #mask
        src_key_mask,src_key_mask_original=self.generate_square_subsequent_mask(src_key_tmp.size(2))
        src_key_tmp=torch.matmul(src_key_tmp,src_key_mask)
        # src_key_tmp_1 = torch.matmul(src_key_tmp, src_key_mask_1)###_1

        src_key_tmp_original = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()  # torch.Size([32, 48, 40])
        src_key_tmp_original = torch.matmul(src_key_tmp_original, src_key_mask_original)

        src_query_mask,src_query_mask_original = self.generate_square_subsequent_mask(src_query_tmp.size(2))
        src_query_tmp = torch.matmul(src_query_tmp, src_query_mask)
        # src_query_tmp_1 = torch.matmul(src_query_tmp, src_query_mask_1)###_1

        src_query_tmp_original = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()  # torch.Size([32, 48, 10])
        src_query_tmp_original = torch.matmul(src_query_tmp_original, src_query_mask_original)
        ####

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)#(20, 20)

        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        vn = input_n - self.kernel_size - output_n + 1#31
        vl = self.kernel_size + output_n#20
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)#(31, 20)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])#torch.Size([992, 20, 48])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [32,40,66*11],torch.Size([32, 31, 960])//torch.Size([32, 16, 960])

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n#20
        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0)#torch.Size([32, 256, 31])//torch.Size([32, 256, 16])
        ##########original
        key_tmp_original = self.convK(src_key_tmp_original / 1000.0)
        # key_tmp_1 = self.convK(src_key_tmp_1 / 1000.0)###_1


        for i in range(itera):

            ###############
            query_tmp = self.convQ(src_query_tmp / 1000.0)#torch.Size([32, 256, 1])
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15#torch.Size([32, 1, 31])

            # query_tmp_1 = self.convQ(src_query_tmp_1 / 1000.0) ###_1
            # score_tmp_1 = torch.matmul(query_tmp_1.transpose(1, 2), key_tmp_1) + 1e-15###_1
            ##########original
            query_tmp_original = self.convQ(src_query_tmp_original / 1000.0)  # torch.Size([32, 256, 1])
            score_tmp_original = torch.matmul(query_tmp_original.transpose(1, 2),
                                            key_tmp_original) + 1e-15  # torch.Size([32, 1, 31])
            score=score_tmp+score_tmp_original

            att_tmp = (score) / (torch.sum(score, dim=2, keepdim=True))#torch.Size([32, 1, 31])
            # att_tmp_numpylook = att_tmp.cpu().data.numpy()  ###_1
            # att_tmp_1 = score_tmp_1 / (torch.sum(score_tmp_1, dim=2, keepdim=True))###_1
            # att_tmp_1_numpylook = att_tmp_1.cpu().data.numpy()###_1
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])#torch.Size([32, 48, 20])

            ##########
            input_gcn = src_tmp[:, idx]#torch.Size([32, 20, 48])//torch.Size([32, 35, 48])
            dct_in_tm = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)#torch.Size([32, 48, 20])
            dct_in_tmp = torch.cat([dct_in_tm, dct_att_tmp], dim=-1)#torch.Size([32, 48, 40])
            dct_out_tmp = self.gcn(dct_in_tmp)#torch.Size([32, 48, 40])
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))#torch.Size([32, 20, 48])

            outputs.append((out_gcn).unsqueeze(2))#torch.Size([32, 20, 1, 48])
            ###############

            if itera > 1:
                # update key-value query
                out_tmp = out_gcn.clone()[:, 0 - output_n:]#torch.Size([32, 10, 48])//test,torch.Size([32, 25, 48])
                src_tmp = torch.cat([src_tmp, out_tmp], dim=1)#torch.Size([32, 60, 48])#torch.Size([32, 70, 60])//torch.Size([32, 75, 48])

                vn = 1 - 2 * self.kernel_size - output_n#-29//-44
                vl = self.kernel_size + output_n#20//35
                idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                          np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)#(10, 20)//(10, 35)

                ########################################################################
                src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)#torch.Size([32, 48, 19])//torch.Size([32, 48, 34])

                # mask
                src_key_mask,src_key_mask_original = self.generate_square_subsequent_mask(src_key_tmp.size(2))
                src_key_tmp = torch.matmul(src_key_tmp, src_key_mask)#torch.Size([32, 48, 19])//torch.Size([32, 48, 34])
                ####

                key_new = self.convK(src_key_tmp / 1000.0)#torch.Size([32, 256, 10])//torch.Size([32, 256, 25])
                key_tmp = torch.cat([key_tmp, key_new], dim=2)#torch.Size([32, 256, 41])#torch.Size([32, 256, 51])//torch.Size([32, 256, 41])
                ###keep 
                key_tmp = key_tmp#[:,:,10:]#  //torch.Size([32, 256, 31])

                src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                    [bs * self.kernel_size, vl, -1])#torch.Size([320, 20, 48])//torch.Size([320, 35, 48])
                src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                    [bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape(
                    [bs, self.kernel_size, -1])#torch.Size([32, 10, 960])
                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)#torch.Size([32, 41, 960])#torch.Size([32, 51, 960])//torch.Size([32, 26, 960])
                ###keep 
                src_value_tmp = src_value_tmp#[:, 10:, :]#  //torch.Size([32, 16, 960])


                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)##torch.Size([32, 60, 10])

                # mask
                src_query_mask,src_query_mask_original = self.generate_square_subsequent_mask(src_query_tmp.size(2))
                src_query_tmp = torch.matmul(src_query_tmp, src_query_mask)#torch.Size([32, 60, 10])

                ########################################################################original
                # mask
                src_key_tmp_original = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)
                # src_key_mask = self.generate_square_subsequent_mask(src_key_tmp.size(2))
                src_key_tmp_original = torch.matmul(src_key_tmp_original,
                                           src_key_mask_original)  # torch.Size([32, 48, 19])//torch.Size([32, 48, 34])
                ####

                key_new_original = self.convK(src_key_tmp_original / 1000.0)  # torch.Size([32, 256, 10])//torch.Size([32, 256, 25])
                key_tmp_original = torch.cat([key_tmp_original, key_new_original],
                                    dim=2)  # torch.Size([32, 256, 41])#torch.Size([32, 256, 51])//torch.Size([32, 256, 41])
                ###keep 
                key_tmp_original = key_tmp_original#[:, :, 10:]  # //torch.Size([32, 256, 31])

                src_query_tmp_original = src_tmp[:, -self.kernel_size:].transpose(1, 2)  ##torch.Size([32, 60, 10])

                # mask
                # src_query_mask = self.generate_square_subsequent_mask(src_query_tmp.size(2))
                src_query_tmp_original = torch.matmul(src_query_tmp_original, src_query_mask_original)  # torch.Size([32, 60, 10])
                ####


        outputs = torch.cat(outputs, dim=2)#torch.Size([32, 20, 1, 48])
        return outputs
