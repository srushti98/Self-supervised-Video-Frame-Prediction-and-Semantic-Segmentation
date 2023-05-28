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
import wandb

class Exp:
    def __init__(self, args, wandb_config):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.wandb_config = wandb_config
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
            device = torch.device('cuda')
            print_log('Use GPU {}:'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
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
                           args.hid_T, args.N_S, args.N_T)
        
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model) #, device_ids=None will take in all available devices
            print(f"Using {torch.cuda.device_count()} GPUs!")
        
        self.model.to(self.device)

        # print(self.model.device_ids)


    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        #self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.wandb_config.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.wandb_config.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def train(self, args):
        config = args.__dict__
        best_model_path = self.path + '/' + 'checkpoint.pth' # load saved model to restart from previous best model (lowest val loss) checkpoint
        
        recorder = Recorder(verbose=True)
        if os.path.isfile(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)
            count = 0

            for batch_x, batch_y in train_pbar:
                #print(count)
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)
                
                #pred_y_norm = pred_y / 255.0
                #batch_y_norm = batch_y / 255.0
                
                loss = self.criterion(pred_y, batch_y)
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                count = count + 1
                torch.cuda.empty_cache()
                #if count == 50: 
                #    break
            train_loss = np.average(train_loss)

            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)
                    torch.cuda.empty_cache()
                    #if epoch % (args.log_step * 100) == 0:
                    self._save(name=str(epoch))
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
                    epoch + 1, train_loss, vali_loss))
                recorder(vali_loss, self.model, self.path)

                wandb.log({'train_loss': train_loss, 'vali_loss': vali_loss, 'epoch': epoch})


        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))
            
            #pred_y_norm = pred_y / 255.0
            #batch_y_norm = batch_y / 255.0
                
            loss = self.criterion(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        folder_path = self.path+'/results/{}/sv/'.format('Debug')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mse, mae, ssim, psnr = metric(preds, trues, 0, 1, True)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        for np_data in ['trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        

        for batch, (trues, preds) in enumerate(zip(trues, preds)):
            for i in range(trues.shape[0]):
                true_img = np.moveaxis(trues[i, :, :, :], 0, -1)
                pred_img = np.moveaxis(preds[i, :, :, :], 0, -1)
                if (batch+1)%100 == 0:
                    #print("Index: ", i)
                    #print("pred img:", pred_img)
                    #print("true img", true_img)
                    wandb.log({f'vali_trues_{batch}_{i+1}': wandb.Image(true_img),
                       f'vali_preds_{batch}_{i+1}': wandb.Image(pred_img)})

        wandb.log({'vali_mse': mse, 'vali_mae': mae, 'vali_ssim': ssim, 'vali_psnr': psnr})


        self.model.train()
        return total_loss