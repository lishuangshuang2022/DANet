from utils import h36motion as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import h5py
import torch.optim as optim


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = 60#opt.in_features  # 48
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.cuda()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    print('>>> loading datasets')

    if not opt.is_eval:
        # dataset = datasets.Datasets(opt, split=0)
        # actions = ["walking", "eating", "smoking", "discussion", "directions",
        #            "greeting", "phoning", "posing", "purchases", "sitting",
        #            "sittingdown", "takingphoto", "waiting", "walkingdog",
        #            "walkingtogether"]
        dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0,
                                 pin_memory=True)  # 1114
        valid_dataset = datasets.Datasets(opt, split=1)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)  # 376

    test_dataset = datasets.Datasets(opt, split=2)  # 120
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)  # 8

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
        # print('testing error: {:.3f}'.format(ret_test['m_ang_h36']))
    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_ang_h36']))
            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt,
                                  epo=epo)
            print('validation error: {:.3f}'.format(ret_valid['m_ang_h36']))
            ret_test = run_model(net_pred, is_train=32, data_loader=test_loader, opt=opt,
                                 epo=epo)
            print('testing error: {:.3f}'.format(ret_test['#1']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_ang_h36'] < err_best:
                err_best = ret_valid['m_ang_h36']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_ang_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_ang = 0
    if is_train <= 1:
        m_ang_seq = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_ang_seq = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dim_used = np.array([6,  7,  8,  9,          12, 13, 14, 15,         21, 22, 23,
                         24,         27, 28, 29, 30,         36, 37, 38, 39, 40, 41,
                         42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57,
                         60, 61, 62, 75, 76, 77, 78, 79, 80, 81,         84, 85, 86])  # 48
    dim_used_1=np.array([6,  7,   8, 9,  10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23,
                         24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41,
                         42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                         60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86])  # 60
    dim_used_1_dil=np.array([4,5,10,11,16,17,22,23,43,44,55,56])
    dim_used_1_res = np.array([0,1,2,3,6,7,8,9,12,13,14,15,
                               18,19,20,21,24,25,26,27,28,29,30,
                               31,32,33,34,35,36,37,38,39,40,41,42,
                               45,46,47,48,49,50,51,52,53,54,57,58,59])
    dim_notused_1 = np.array([0, 1, 2, 3, 4, 5, 10, 11, 16, 17, 18, 19, 20, 25, 26, 31, 32, 33, 34, 35,
                              48, 49, 50, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 82, 83,
                              87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98])  # 99-48
    seq_in = opt.kernel_size

    itera = 1
    idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
            out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    st = time.time()
    for i, (ang_h36) in enumerate(data_loader):
        batch_size, seq_n, _ = ang_h36.shape  # 32,60,99
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        ang_h36 = ang_h36.float().cuda()  # 32,60,99
        ang_sup = ang_h36.clone()[:, :, dim_used_1][:, -out_n - seq_in:]  # torch.Size([32, 20, dim=60])
        # ang_sup[:, :,dim_used_1_dil] = 0

        ang_src = ang_h36.clone()[:, :, dim_used_1]  # torch.Size([32, 60, dim=60])
        # ang_src[:, :,dim_used_1_dil] = 0
        ang_src_numpylook = ang_src.cpu().data.numpy()
        ang_out_all = net_pred(ang_src, output_n=out_n, itera=itera, input_n=in_n)  # torch.Size([32, 20, 1, dim=60])
        ang_out_all_numpylook=ang_out_all.cpu().data.numpy()

        ang_out = ang_h36.clone()[:, in_n:in_n + out_n]  # torch.Size([32, 10, 99])
        ang_out[:, :, dim_used_1] = ang_out_all[:, seq_in:, 0]  # torch.Size([32, 10, 99])


        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:
            loss_ang = torch.mean(torch.sum(torch.abs(ang_out_all[:, :, 0] - ang_sup), dim=2))
            loss_all = loss_ang
            optimizer.zero_grad()
            loss_all.backward()
            grad_norm = nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()

            # update log values
            l_ang += loss_ang.cpu().data.numpy() * batch_size

        if is_train <= 1:  # if is validation or train simply output the overall mean error
            with torch.no_grad():
                ang_out_euler = ang_out.reshape([-1, 99]).reshape([-1, 3])
                ang_gt_euler = ang_h36[:, in_n:in_n + out_n].reshape([-1, 99]).reshape([-1, 3])

                import utils.data_utils as data_utils
                ang_out_euler = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(ang_out_euler))
                ang_out_euler = ang_out_euler.view(-1, 99)
                ang_gt_euler = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(ang_gt_euler))
                ang_gt_euler = ang_gt_euler.view(-1, 99)

                eulererr_ang_seq = torch.mean(torch.norm(ang_out_euler - ang_gt_euler, dim=1))

            m_ang_seq += eulererr_ang_seq.cpu().data.numpy() * batch_size
        else:

            with torch.no_grad():
                ang_out_euler = ang_out.reshape([-1, 99]).reshape([-1, 3])
                ang_gt_euler = ang_h36[:, in_n:in_n + out_n].reshape([-1, 99]).reshape([-1, 3])

                import utils.data_utils as data_utils
                ang_out_euler = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(ang_out_euler))
                ang_out_euler = ang_out_euler.view(-1, out_n, 99)
                ang_gt_euler = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(ang_gt_euler))
                ang_gt_euler = ang_gt_euler.view(-1, out_n, 99)

                eulererr_ang_seq = torch.sum(torch.norm(ang_out_euler - ang_gt_euler, dim=2), dim=0)
            m_ang_seq += eulererr_ang_seq.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))
    ret = {}
    if is_train == 0:
        ret["l_ang"] = l_ang / n

    if is_train <= 1:
        ret["m_ang_h36"] = m_ang_seq / n
    else:
        m_ang_h36 = m_ang_seq / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_ang_h36[j]
    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
