import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import numpy as np
from logging import getLogger, FileHandler, Formatter
import datetime
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import hashlib

parser = argparse.ArgumentParser(description='model training log')

parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--num_steps', type=int, default=5)# 5 for rollout times
parser.add_argument('--num_frames', type=int, default=1000000)# 10e5 , 100万
parser.add_argument('--num_ics', type=int, default=1)# default:1 for imagination core
parser.add_argument('--mode', type=str, default='regular')
parser.add_argument('--learning_rate', type=float, default=7e-4)
parser.add_argument('--free_model', type=str, default='dummy')
parser.add_argument('--env_model', type=str, default='dummy')

def get_my_logger(label, args):
    logger = getLogger('training log')
    # ログレベルの設定（2）
    logger.setLevel(10)
    # ログのファイル出力先を設定（4）
    fh = FileHandler('./{}_training_log_{}.log'.format(label, now_time()))
    formatter = Formatter('%(asctime)s|%(levelname)s|%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.log(20, 'params_str:'+str(dict(vars(args))))
    return logger

def argstr(args):
    'argsを受け取って、文字列にする、これを安直に変えないほうが良い'
    argd = vars(args)
    argstring = '-'.join([str(k)+'：'+str(argd[k]) for k in argd])
    return argstring

def hash_arg(args):
    '受け取ったパラメータをハッシュ化した文字列にする'
    hashed = hashlib.md5(argstr(args).encode())
    return hashed.hexdigest()


def params_logger():
    'パラメータとハッシュ化された名前を記録するためのlogger'
    logger = getLogger('to save training parmas and taged name')
    # ログレベルの設定（2）
    logger.setLevel(10)
    # ログのファイル出力先を設定（4）
    fh = FileHandler('./parames.log')
    formatter = Formatter('%(asctime)s|%(levelname)s|%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


USE_CUDA = torch.cuda.is_available()
#Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
def Variable(x, volatile=False):
    if volatile:
        with torch.no_grad():
            if USE_CUDA:
                x = x.cuda()
        return x
    if USE_CUDA:
        return torch.autograd.Variable(x).cuda()
    return torch.autograd.Variable(x)

def save_model(model, label, args):
    torch.save(model.state_dict(), './trained_models/{}_{}_{}'.format(label, hash_arg(args), now_time()))

def now_time():
    # to convert to JST time
    now = datetime.datetime.utcnow()+datetime.timedelta(hours=9)
    return now.strftime('%Y%m%d-%H.%M.%S')

def new_writer(label, args):
    comment = '{}_{}'.format(label, hash_arg(args))
    logdir = '/tblog/'+comment+'/'+now_time()
    writer = SummaryWriter(log_dir=logdir, comment=comment)
    return writer

def write_histograms(model, writer, step):
    model_dict = model.state_dict()
    for key in model_dict:
        writer.add_histogram(model._get_name()+':'+key, model_dict[key], step)

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init
