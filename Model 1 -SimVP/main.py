import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')
import wandb

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--data_root', default='/scratch/ad6489/dlproject/frame-pred/dataset/unlabeled')
    parser.add_argument('--dataname', default='moving_objects', choices=['moving_objects'])
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[11,3,160,240], type=int,nargs='*')
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




def main_module():
    args = create_parser().parse_args()
    config = args.__dict__

    wandb.init(project="frame-pred-simvp", config=config)


    exp = Exp(args, wandb.config)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    wandb.log({"test_final_mse": mse})

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    wandb.finish()


if __name__ == '__main__':

    sweep_configuration = {
        "method": "random",
        "metric": { 
            "name": "vali_loss",
            "goal": "minimize"
        },
        "parameters":{
            "lr": {"values": [1e-3, 1e-2]}
            }
        }

    # Start the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='frame-pred-simvp', 
        )

    wandb.agent(sweep_id, function=main_module, count=2)


