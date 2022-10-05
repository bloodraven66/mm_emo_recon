from . import iemocap_loader
from torch.utils.data import DataLoader

def loaders(args):
    loaders_ = []
    for mode in ['train', 'valid', 'test']:
        dataset = iemocap_loader.iemocap_DATASET(mode, args)
        exit()
        # collate_fn = iemocap_loader.iemocap_Collate()
        shuffle=True if mode=='train' else False
        loaders_.append(DataLoader(dataset, num_workers=1, shuffle=shuffle,
                          batch_size=args.batch_size, pin_memory=False,
                          drop_last=True, collate_fn=collate_fn))
    return loaders_