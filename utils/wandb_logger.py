import wandb
import matplotlib.pyplot as plt

class WandbLogger():
    def __init__(self, config):
        self.run = wandb.init(reinit=True,
                            name='_'.join([config.fastspeech, config.data.name, config.exp_name]),
                            project=config.wandb_project,
                            config=config,
                            notes=config.notes,
                            tags=config.tags)

    def log(self, dct):
        wandb.log(dct)

    def log_plots(self, gnd, pred):
        fig, ax = plt.subplots(2, 4, figsize=(12, 4))
        for j in range(4):
            ax[0][j].imshow(gnd[j])
            ax[1][j].imshow(pred[j].T)
        plt.tight_layout()
        wandb.log({"mels": wandb.Image(plt)})

    def summary(self, dct):
        for key in dct:
            wandb.run.summary[key] = dct[key]

    def end_run(self):
        self.run.finish()

    def log_audio(Self, aud, name='val'):
        wandb.log({name: wandb.Audio(aud,  sample_rate=22050)})