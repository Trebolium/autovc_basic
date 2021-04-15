import os, pdb, pickle, argparse, shutil
from solver_encoder import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def overwrite_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    vcc_loader = get_loader(config)
    # pass dataloader and configuration params to Solver NN
    solver = Solver(vcc_loader, config)
    solver.train()
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=16)
    parser.add_argument('--one_hot', type=str2bool, default=False, help='Toggle 1-hot mode')
    parser.add_argument('--shape_adapt', type=str2bool, default=True, help='adjust shapes of tensors to match automatically')
    parser.add_argument('--which_cuda', type=int, default=0, help='Determine which cuda to use')
    parser.add_argument('--data_dir', type=str, default='/homes/bdoc3/my_data/autovc_data/autovc_basic', help='path to config file to use')    
    # Training configuration.
    parser.add_argument('--file_name', type=str, default='defaultSetup')
    parser.add_argument('--spmel_dir', type=str, default='/homes/bdoc3/my_data/autovc_data/spmel')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--adam_init', type=float, default=0.0001, help='Define initial Adam optimizer learning rate')
    parser.add_argument('--train_size', type=int, default=20, help='Define how many speakers are used in the training set')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    parser.add_argument('--psnt_loss_weight', type=float, default=1.0, help='Determine weight applied to postnet reconstruction loss')
    parser.add_argument('--prnt_loss_weight', type=float, default=1.0, help='Determine weight applied to pre-net reconstruction loss')
 
    # Miscellaneous.
    parser.add_argument('--load_ckpts', type=str, default='', help='toggle checkpoint load function')
    parser.add_argument('--ckpt_freq', type=int, default=50000, help='frequency in steps to mark checkpoints')
    parser.add_argument('--spec_freq', type=int, default=10000, help='frequency in steps to print reconstruction illustrations')
    parser.add_argument('--log_step', type=int, default=10)
    config = parser.parse_args()
    if config.one_hot==True:
        config.dim_emb=config.train_size
    
    print(config)
    # pdb.set_trace()
    overwrite_dir(config.data_dir +'/model_saves/' +config.file_name)
    os.makedirs(config.data_dir +'/model_saves/' +config.file_name +'/ckpts')
    os.makedirs(config.data_dir +'/model_saves/' +config.file_name +'/generated_wavs')
    os.makedirs(config.data_dir +'/model_saves/' +config.file_name +'/image_comparison')
    with open(config.data_dir +'/model_saves/' +config.file_name +'/config.pkl', 'wb') as config_file:
        pickle.dump(config, config_file)

    main(config)
