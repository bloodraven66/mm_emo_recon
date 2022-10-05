from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np
import torch
import re
from utils.logger import logger
import random
import functools
import torchaudio
from scipy import ndimage
import json
import pickle
import shutil
class iemocap_DATASET(Dataset):
    def __init__(self, mode, args):
        if not os.path.exists(os.path.join('manifest', args.use_data.name, args.use_data.save_manifest+str(args.use_data.test_spk)+mode+'.json')):
            self.init_files(args)

    def init_files(self, args):
        speaker_dict = {str(i + 1): [] for i in range(10)}

        speaker_count = 0
        for k in range(5):
            session = self.load_session("%s%s" % (args.use_data.path, k + 1))
            for idx in range(len(session)):
                if session[idx][2] == "F":
                    speaker_dict[str(speaker_count + 1)].append(session[idx])
                else:
                    speaker_dict[str(speaker_count + 2)].append(session[idx])
            speaker_count += 2
        test_spk_id = args.use_data.test_spk
        # def get_files(path, extension='.wdseg'):
        #     if isinstance(path, str): path = Path(path).expanduser().resolve()
        #     return list(path.rglob(f'*{extension}'))    
        # words = {w.stem:str(w) for w in get_files('../other_tts_data/IEMOCAP_full_release/')}
        # text = {}
        # wavscp = {}
        # utt2spk = {}
        # for s in speaker_dict:
        #     spk = f'spk{s}'
        #     print((len(speaker_dict[s])))
        #     wavscp_ = {spk+'-'+Path(l[-3]).stem : 'wavs/'+Path(l[-3]).stem+'.wav' for l in speaker_dict[s]}
        #     utt2spk_ = {w:w.split('-')[0] for w in wavscp_}
        #     for ss in wavscp_:
        #         with open(words['-'.join(ss.split('-')[1:])] ,'r') as f:
        #             data = f.read().split('\n')
        #         data = re.sub("[\(\[].*?[\)\]]", "", ' '.join([d.split()[-1].strip() for d in data[1:] if '\t' in d]))
        #         data = data.replace('<s>', '')
        #         data = data.replace('</s>', '')
        #         data = data.replace('<sil>', '')
        #         data = re.sub(' +', ' ', data).strip()
        #         text[ss] = data
        #     for l in wavscp_:
        #         wavscp[l] = wavscp_[l]
        #     for l in utt2spk_:
        #         utt2spk[l] = utt2spk_[l]
        #         # exit()
        #     # for ss in wavscp:
        #         # shutil.copy2(wavscp[ss], '../other_tts_data/IEMOCAP_full_release/wavs/'+Path(wavscp[ss]).stem+'.wav')
        # with open('manifest/data/text', 'w') as f:
        #     for l in text:
        #         f.write(l + ' ' + text[l] + '\n')
        # with open('manifest/data/wavscp', 'w') as f:
        #     for l in wavscp:
        #         f.write(l + ' ' + wavscp[l] + '\n')
        # with open('manifest/data/utt2spk', 'w') as f:
        #     for l in utt2spk:
        #         f.write(l + ' ' + utt2spk[l] + '\n')
        # exit()
        data_split = {k: [] for k in ["train", "valid", "test"]}
        data_split["test"].extend(speaker_dict[str(test_spk_id)])

        # use the speaker in the same session as validation set
        if test_spk_id % 2 == 0:
            valid_spk_num = test_spk_id - 1
        else:
            valid_spk_num = test_spk_id + 1

        data_split["valid"].extend(speaker_dict[str(valid_spk_num)])

        for i in range(1, 11):
            if i != valid_spk_num and i != test_spk_id:
                data_split["train"].extend(speaker_dict[str(i)])
        for split in ['train', 'test', 'valid']:
            json_dict = {}
            for obj in data_split[split]:
                wav_file = obj[0]
                emo = obj[1]
                # Read the signal (to retrieve duration in seconds)
                audio, _ = torchaudio.load(wav_file)
                signal = audio.transpose(0, 1).squeeze(1)
                # signal = read_audio(wav_file)
                duration = signal.shape[0] / 16000


                uttid = wav_file.split("/")[-1][:-4]

                # Create entry for this utterance
                json_dict[uttid] = {
                    "wav": wav_file,
                    "length": duration,
                    "emo": emo,
                }
            json_file_folder = os.path.join('manifest', args.use_data.name)
            if not os.path.exists(json_file_folder): os.mkdir(json_file_folder)
            json_file = os.path.join(json_file_folder, args.use_data.save_manifest+str(args.use_data.test_spk)+split+'.json')
            # Writing the dictionary to the json file
            with open(json_file, mode="w") as json_f:
                json.dump(json_dict, json_f, indent=2)

            logger.info(f"{json_file} successfully created!")
        spk_data = {}
        for spk in speaker_dict:
            spk_data[spk] = {}
            for utt in speaker_dict[spk]:
                if utt[-2] not in spk_data[spk]:
                    spk_data[spk][utt[-2]] = 1
                else:
                    spk_data[spk][utt[-2]] += 1
        for spk in speaker_dict:
            print(spk)
            for l in ['ang', 'hap', 'sad', 'neu']:
                print(l, spk_data[spk][l], end='\t')
            print()
        print(spk_data)

    def load_session(self, pathSession):
        
        pathEmo = pathSession + "/dialog/EmoEvaluation/"
        pathWavFolder = pathSession + "/sentences/wav/"

        improvisedUtteranceList = []
        for emoFile in [
            f
            for f in os.listdir(pathEmo)
            if os.path.isfile(os.path.join(pathEmo, f))
        ]:
            for utterance in self.load_utterInfo(pathEmo + emoFile):
                if utterance == []: continue
                if (
                    (utterance[3] == "neu")
                    or (utterance[3] == "hap")
                    or (utterance[3] == "sad")
                    or (utterance[3] == "ang")
                    # or (utterance[3] == "exc")
                ):
                    path = (
                        pathWavFolder
                        + utterance[2][:-5]
                        + "/"
                        + utterance[2]
                        + ".wav"
                    )

                    label = utterance[3]
                    if label == "exc":
                        label = "hap"

                    if emoFile[7] != "i" and utterance[2][7] == "s":
                        improvisedUtteranceList.append(
                            [path, label, utterance[2][18]]
                        )
                    else:
                        improvisedUtteranceList.append(
                            [path, label, utterance[2][15]]
                        )
        return improvisedUtteranceList
        
    def __len__(self):
        pass

    def __getitem__(self, i):
        pass

    def load_utterInfo(self, inputFile):
        """
        Load utterInfo from original IEMOCAP database
        """

        # this regx allow to create a list with:
        # [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
        # [V, A, D] means [Valence, Arousal, Dominance]
        pattern = re.compile(
            "[\[]*[0-9]*[.][0-9]*[ -]*[0-9]*[.][0-9]*[\]][\t][a-z0-9_]*[\t][a-z]{3}[\t][\[][0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[\]]",
            re.IGNORECASE,
        )  # noqa
        if Path(inputFile).stem.startswith('.'): return [] 
        with open(inputFile, "r") as myfile:
            
            data = myfile.read().replace("\n", " ")
        result = pattern.findall(data)
        out = []
        for i in result:
            a = i.replace("[", "")
            b = a.replace(" - ", "\t")
            c = b.replace("]", "")
            x = c.replace(", ", "\t")
            out.append(x.split("\t"))
        return out

  



class TTS_Collate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, ):
        pass

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """

        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        dur_padded = torch.LongTensor(len(batch), max_input_len)
        dur_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            dur = batch[ids_sorted_decreasing[i]][2]
            dur_padded[i, :dur.size(0)] = dur

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.IntTensor(len(batch))
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        vals = torch.LongTensor(len(batch))
        use_vals = False
        filenames = []
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i] = mel.size(1)
            output_lengths[i] = mel.size(1)
            filenames.append(batch[ids_sorted_decreasing[i]][3])
            if len(batch[ids_sorted_decreasing[0]]) == 5:
                vals[i] = batch[ids_sorted_decreasing[0]][-1]
                use_vals = True

        # attn_prior_padded = torch.zeros(len(batch), max_target_len,
        #                                 max_input_len)
        # attn_prior_padded.zero_()
        # for i in range(len(ids_sorted_decreasing)):
        #     prior = batch[ids_sorted_decreasing[i]][2]
        #     attn_prior_padded[i, :prior.size(0), :prior.size(1)] = prior
        if use_vals:
            return text_padded, input_lengths, mel_padded, gate_padded, filenames,\
                dur_padded, vals
        return text_padded, input_lengths, mel_padded, gate_padded, filenames,\
            dur_padded