
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics # part of pytorch linghtning with all the metrics
import wandb
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix

class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-2):
        super(ColaModel, self).__init__()
        self.num_classes = 2
        self.train_accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=self.num_classes)
        self.val_accuracy_metric = torchmetrics.Accuracy(task="multiclass",num_classes=self.num_classes)
        self.f1_metric = torchmetrics.F1Score(task="multiclass",num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", task="multiclass",num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", task="multiclass",num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro", task="multiclass", num_classes=self.num_classes)
        self.recall_micro_metric = torchmetrics.Recall(average="micro", task="multiclass", num_classes=self.num_classes)

        self.save_hyperparameters()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.W = nn.Linear(self.bert.config.hidden_size, 2)
        self.validation_step_outputs = []        

    # def forward(self, input_ids, attention_mask):
    #     outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

    #     h_cls = outputs.last_hidden_state[:, 0]
    #     logits = self.W(h_cls)
    #     return logits

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    # def training_step(self, batch, batch_idx):
    #     logits = self.forward(batch["input_ids"], batch["attention_mask"])
    #     loss = F.cross_entropy(logits, batch["label"])
    #     self.log("train_loss", loss, prog_bar=True)
    #     preds = torch.argmax(logits.logits, 1)
    #     return loss

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        # loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    # def validation_step(self, batch, batch_idx):
    #     logits = self.forward(batch["input_ids"], batch["attention_mask"])
    #     loss = F.cross_entropy(logits, batch["label"])
    #     _, preds = torch.max(logits, dim=1)
    #     val_acc = accuracy_score(preds.cpu(), batch["label"].cpu())
    #     val_acc = torch.tensor(val_acc)
    #     self.log("val_loss", loss, prog_bar=True)
    #     self.log("val_acc", val_acc, prog_bar=True)

    # def validation_step(self, batch, batch_idx):
    #     labels = batch["label"]
    #     outputs = self.forward(
    #         batch["input_ids"], batch["attention_mask"], labels=batch["label"]
    #     )
    #     preds = torch.argmax(outputs.logits, 1)

    #     # Metrics
    #     valid_acc = self.val_accuracy_metric(preds, labels)
    #     precision_macro = self.precision_macro_metric(preds, labels)
    #     recall_macro = self.recall_macro_metric(preds, labels)
    #     precision_micro = self.precision_micro_metric(preds, labels)
    #     recall_micro = self.recall_micro_metric(preds, labels)
    #     f1 = self.f1_metric(preds, labels)

    #     # Logging metrics
    #     self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
    #     self.log("valid/acc", valid_acc, prog_bar=True)
    #     self.log("valid/precision_macro", precision_macro, prog_bar=True)
    #     self.log("valid/recall_macro", recall_macro, prog_bar=True)
    #     self.log("valid/precision_micro", precision_micro, prog_bar=True)
    #     self.log("valid/recall_micro", recall_micro, prog_bar=True)
    #     self.log("valid/f1", f1, prog_bar=True)
    #     return {"labels": labels, "logits": outputs.logits}

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        
        self.validation_step_outputs.append({"labels": labels, "logits": outputs.logits})

        return {"labels": labels, "logits": outputs.logits}


    # def validation_epoch_end(self, outputs):
    #     labels = torch.cat([x["labels"] for x in outputs])
    #     logits = torch.cat([x["logits"] for x in outputs])
    #     preds = torch.argmax(logits, 1)

    #     cm = confusion_matrix(labels.numpy(), preds.numpy())
    
    def on_validation_epoch_end(self):
        # Replace validation_epoch_end with on_validation_epoch_end
        outputs = self.validation_step_outputs
        
        # Clear the outputs list for the next epoch
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)

        # Wandb logging
        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.cpu().numpy(), y_true=labels.cpu().numpy()
                )
            }
        )
        
        # Reset the outputs list for next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])