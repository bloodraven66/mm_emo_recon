from attrdict import AttrDict
import yaml
import pickle
import os, sys
from multiprocessing import Pool
from tqdm import tqdm
import librosa
import random
from pathlib import Path
import torch
import logging
import numpy as np
from data_prep import data_handler
from trainer import iemocap_trainer
from models import fs
from utils.logger import logger


def load_config(yamlFile):
    with open(yamlFile) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        cfg = AttrDict(config)
    return cfg


def load_mode(args):

    loaders = data_handler.loaders(args)
    trainer = get_trainer(args)
    if args.infer: 
        return loaders, trainer, None, args
    model, args = get_model(args)
    args = AttrDict(args)
    return loaders, trainer, model, args


def get_model(args, custom_config=False, custom_args=False):
    with open(args.data.symbol_dict, 'rb') as f:
        data =  pickle.load(f)
    if args.data.name in ['iemocap']:
        
        if args.model == 'fastspeech':
            fs_args = load_config(args.fs_config)
            model = fs.FastSpeech(n_mel_channels=args.data.n_mels,
                                                    n_symbols=len(data)+1, **fs_args)
       
            
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
    args = {**args, **fs_args}
    return model, args

def get_trainer(args):
    if args.data.name == 'ljspeech':
        if args.model == 'fastspeech':
            trainer =  fs_tts_trainer
        elif args.model == 'fastspeech_c':
            trainer =  fs_tts_trainer_c
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return trainer