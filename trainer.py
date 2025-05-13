from data import DataModule
from model import ColaModel

import torch 
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

cola_data = DataModule()
cola_model = ColaModel()

# checkpoint_callback = ModelCheckpoint(
#     dirpath="./models", monitor="val_loss", mode="min"
# ) # Callback: This will save the trained model. We can selectively choose which model to save by monitoring a metric.

# early_stopping_callback = EarlyStopping(
#     monitor="val_loss", patience=3, verbose=True, mode="min"
# ) # Callback:  helps the model not to overfit by mointoring on a certain parameter (val_loss in this case)

wandb_logger = WandbLogger(project="MLOps Basics")

trainer = pl.Trainer(
    logger=wandb_logger,
    default_root_dir="logs",
    # gpus=(1 if torch.cuda.is_available() else 0),
    max_epochs=3,
    fast_dev_run=True, # will run one batch of training step and one batch of validation step to make sure everything is setup properly
    # logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
    # callbacks=[checkpoint_callback, early_stopping_callback],
    )

trainer.fit(cola_model, cola_data)