import os, sys
from utils import common

def main():
    cfg = common.load_config(configfile)
    loaders, trainer, model, cfg = common.load_mode(cfg)
    trainer.main(cfg, model, loaders)

if __name__ == '__main__':
    configfile = sys.argv[1]
    main()