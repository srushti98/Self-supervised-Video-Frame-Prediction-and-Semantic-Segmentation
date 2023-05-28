import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from model import SimVP
from tqdm import tqdm
from API import *
from utils import *

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        #self._select_optimizer()
        self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)

        self.model1_path=self.args.model1_path

        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)

    def _get_data(self):
        config = self.args.__dict__
        self.vali_loader,self.data_mean, self.data_std = load_data(**config)

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def prediction_eval(self,args):

        self.model.load_state_dict(torch.load(self.model1_path))
        self.model.eval()
        inputs_lst, preds_lst ,last_frame_list= [], [],[]
        vali_pbar = tqdm(self.vali_loader)

        for i,batch_x in enumerate(vali_pbar):
            pred_y = self.model(batch_x.to(self.device))
            last_frame_y=pred_y[:,10,:,:,:]
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                batch_x,pred_y,last_frame_y], [inputs_lst, preds_lst,last_frame_list]))
    
        print("pred_y.shape:",pred_y.shape)
        print("last_frame list len:",len(last_frame_list))

        inputs = np.concatenate(inputs_lst, axis=0)
        preds = np.concatenate(preds_lst, axis=0)

        print("inputs and pred concatenated successfully")

        last_frames = np.concatenate(last_frame_list, axis=0)

        print("final last frame:",last_frames.shape)
        print("concatenation complete")

        folder_path = self.path + '/results/{}/sv/'.format(args.ex_name)
        
        print("folder path:",folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print("directory made sucessfully:",folder_path)

        np.save(osp.join(folder_path, 'inputs.npy'),inputs)

        print("inputs saved successfully")
        np.save(osp.join(folder_path, 'preds.npy'),preds)

        print("preds saved successfully")
        np.save(osp.join(folder_path, 'last_frames.npy'),last_frames)
        
        print("last_frames saved successfully")
        print_log("Testing done successfully!")


