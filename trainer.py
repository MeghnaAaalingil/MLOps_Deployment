import torch
import wandb
import hydra
import logging
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from data import DataModule
from model import ColaModel
from omegaconf.omegaconf import OmegaConf

logger = logging.getLogger(__name__)

class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        print("Batch keys:", val_batch.keys())
        sentences = val_batch["sentence"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.cpu().numpy(), "Predicted": preds.cpu().numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )
        
@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")
    # cola_data = DataModule()
    # cola_model = ColaModel()
    #with hydra
    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel(cfg.model.name)
    root_dir = hydra.utils.get_original_cwd()
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{root_dir}/models",
        filename="best-checkpoint",
        monitor="valid/loss",
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )
    accelerator = "auto"
    devices = 1
    wandb_logger = WandbLogger(project="MLOpsPipeline", entity="meghnaaalingil-university-at-buffalo")
    trainer = pl.Trainer(
        #default_root_dir="logs",
        accelerator=accelerator,
        devices=devices,
        max_epochs=cfg.training.max_epochs,
        # fast_dev_run=True,
        #logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
        logger = wandb_logger,
        #callbacks=[checkpoint_callback, early_stopping_callback],
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(cola_data), early_stopping_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
    )
    trainer.fit(cola_model, cola_data)
    wandb.finish()
    

if __name__ == "__main__":
    main()
    #tensorboard --logdir=logs/cola