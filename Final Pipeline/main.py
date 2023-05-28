import argparse
from exp import Exp
from unet import *

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--model1_path', default='/scratch/sxp8182/SimVP-Simpler-yet-Better-Video-Prediction/results_5/Debug/checkpoint.pth', type=str)

    parser.add_argument('--model2_path', default='/scratch/sxp8182/model2504_new.pt', type=str)  # remind to give correct path

    # dataset parameters
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--data_root', default='/scratch/sxp8182/hidden')
    parser.add_argument('--dataname', default='not_mmnist', choices=['not_mmnist', 'taxibj'])
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[11,3,160,240], type=int,nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=512, type=int)
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print("Inside Start")
    exp.prediction_eval(args)

    UNET_Module(args)
