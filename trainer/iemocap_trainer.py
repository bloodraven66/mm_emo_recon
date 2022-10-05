import os, sys
import torch
# from models.fastspeech import FastSpeechLoss
from utils.wandb_logger import WandbLogger
from tqdm import tqdm
os.environ["WANDB_SILENT"] = "true"
import logging
import librosa
from g2p_en import G2p
import pickle

import numpy as np
from utils.logger import logger
class Train_loop():
    def __init__(self, args, model):
        self.args = args
        self.model = model.to(args.device)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        logger.info(f'{pytorch_total_params}')
        self.criterion = FastSpeechLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                        lr=args.lr,
                                        betas=(0.9, 0.98),
                                        eps=1e-9,
                                        weight_decay=float(args.weight_decay))
        self.logger = WandbLogger(args)
        self.logger.log({'full_params':pytorch_total_params})
        self.bestloss = 100
        self.prev_chk = None

    def save_checkpoint(self):
        chk_name = str(self.epoch // self.args.freq)
        name, extn = self.args.chk.split('.')
        chk_name = self.args.model + '_' + name + '_' + chk_name + '.' + extn
        save_path = os.path.join('saved_models', chk_name)
        if not os.path.exists('saved_models'):
            os.mkdir('saved_models')
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.bestloss,
            }, save_path)

    def load_checkpoint(self, chk=None):
        chk = self.args.chk if chk == None else chk
        save_path = 'saved_models'
        if os.path.exists(os.path.join(save_path, chk)):
            checkpoint = torch.load(os.path.join(save_path, chk))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f'Loaded checkpoint {chk}')
        else:
            logger.info('Checkpoint not found')

    def train(self, loader):
        self.model.train()
        epoch_log = {}

        for data in tqdm(loader):
            data = self.to_gpu(data)
            if torch.max(data[0]) > torch.tensor(118):
                logger.info('bad input, skipped batch')
                continue

            out = self.model(data)

            loss, meta = self.criterion(out, (data[2], data[1], data[3], data[-1]))
            if torch.isnan(loss):
                print( meta)
                print('bad loss')
                exit()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1000.0)
            self.optimizer.step()
            for key in meta:
                if key not in epoch_log:
                    epoch_log[key] = [meta[key].detach().cpu().item()]
                else:
                    epoch_log[key].append(meta[key].detach().cpu().item())
            # break
        self.logger.log({'train_epoch_'+m:sum(epoch_log[m])/len(epoch_log[m]) for m in epoch_log})

    def validate(self, loader):
        self.model.eval()
        epoch_log = {}
        for data in loader:
            data = self.to_gpu(data)
            out = self.model(data)

            loss, meta = self.criterion(out, (data[2], data[1], data[3], data[-1]))
            if torch.isnan(loss):
                print('bad loss')
                exit()
            # meta['kl_loss'] = torch.zeros_like(loss)
            for key in meta:
                if key not in epoch_log:
                    epoch_log[key] = [meta[key]]
                else:
                    epoch_log[key].append(meta[key])

        self.logger.log({'val_epoch_'+m:sum(epoch_log[m])/len(epoch_log[m]) for m in epoch_log})
        # if sum(epoch_log['loss'])/len(epoch_log) < self.bestloss:
        self.bestloss = sum(epoch_log['loss'])/len(epoch_log)
        print('Saving checkpoint..')
        self.save_checkpoint()
        self.logger.log_plots(data[2][:self.args.num_samples].detach().cpu().numpy(),
                                out[0][:self.args.num_samples].detach().cpu().numpy())
        aud = self.get_audio(out[0].detach().cpu().numpy()[0],  data[3])
        self.logger.log_audio(aud)
    
    def infer(self):
        sentence = "What are you doing?"
        with open(self.args.data.symbol_dict, 'rb') as f:
            symbols =  pickle.load(f)
        symbols_to_idx = {i:idx+1 for idx, i in enumerate(symbols)}
        g2p = G2p()
        phonemes = g2p(sentence)
        print(phonemes)
        # # phonemes[1]
        # # phonemes = [p.replace('1', '0') for p in phonemes]
        # phonemes[4] = 'AA1'
        
        # phonemes[6:] = phonemes[5:]
        # phonemes[5] = 'AA1'
        # # phonemes[12] = 'IH1'
        # print(phonemes)
        phon_names = '_'.join(phonemes)
        phonemes = [t.replace('1', '').replace('0', '').replace('2', '') for t in phonemes]
        print(phonemes)
        phonemes = [symbols_to_idx[p] for p in phonemes]
        
        print(phonemes)
        
        
        self.model.eval()
        phons = torch.from_numpy(np.array(phonemes))[None, :].to(self.args.device)
        lens = torch.from_numpy(np.array([len(phonemes)]))
        inputs = (phons, lens, None, None, None, None)
        with torch.no_grad():

            out, *_ = self.model(inputs = inputs, infer=True)
        with open('tmp/infer.npy', 'wb') as f:
            np.save(f, out[0].cpu().numpy())
        aud = self.get_audio(out[0].cpu().numpy(), None)
        self.logger.log_audio(aud, name=phon_names)

    def to_gpu(self, batch):
        return [batch[b].to(self.args.device)  if not isinstance(batch[b], list) else batch[b] for b in range(len(batch))]

    def get_audio(self, sample, lengths, index=0):
            MAX_WAV_VALUE = 32768.0
            if lengths is not None:
                y_gen_tst = sample[:int(lengths[index])].T
            else:
                y_gen_tst = sample.T
            y_gen_tst = np.exp(y_gen_tst)
            S = librosa.feature.inverse.mel_to_stft(
                    y_gen_tst,
                    power=2,
                    sr=22050,
                    n_fft=1024,
                    fmin=0,
                    fmax=8000.0)
            audio = librosa.core.griffinlim(
                    S,
                    n_iter=32,
                    hop_length=256,
                    win_length=1024)
            audio = audio * MAX_WAV_VALUE
            audio = audio.astype('int16')
            return audio

def main(args, model, loaders):
    train_loop = Train_loop(args, model)
    if args.infer:
        train_loop.load_checkpoint(chk=args.load_infer)
        train_loop.infer()
        exit()
    if args.load_chk:
        train_loop.load_checkpoint()

    for epoch in range(args.num_epochs):
        train_loop.epoch = epoch
        train_loop.train(loaders[0])
        train_loop.validate(loaders[1])
    train_loop.logger.end_run()