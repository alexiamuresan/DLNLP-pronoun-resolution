import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
import torchmetrics

class GAPCorefClassifier(pl.LightningModule):
    """
    Coreference classifier on GAP dataset using a transformer-based model.
    """

    def __init__(
        self,                      
        lm_name: str='roberta-large',
        num_classes: int=3,
        lr: float = 5e-4,
        weight_decay: float = 5e-4,
        **kwargs,
    ):
        """
        Initialize GAP Coreference Classifier.

        Args:
            lm_name (str): Pretrained model name or path.
            num_classes (int): Number of output labels.
            lr (float): Learning rate for optimizer (Adam).
            weight_decay (float): Weight decay for optimizer (Adam).
        """
        super().__init__()
        self.save_hyperparameters()
        
        # instantiate BERT model
        self.model = AutoModel.from_pretrained(lm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        # self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.model.config.hidden_size, num_classes)
        
        # instantiate loss function
        self.loss = nn.CrossEntropyLoss()

        # instantiate accuracy metric
        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro')
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')

    def forward(self, batch):
        # Extract tokenized inputs from batch
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Obtain model outputs
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Pass through the final classification layer
        logits = self.fc(pooled_output)
        
        return logits

    def step(self, batch):
        labels = batch['label'].to(self.device)
        logits = self.forward(batch)
        
        # Compute loss
        loss = self.loss(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        self.log("train_loss", loss, batch_size=len(y), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds, y)  # Compute accuracy
        f1 = self.f1(preds, y)  # Compute f1
        self.log("val_loss", loss, batch_size=len(y), on_epoch=True)
        self.log("val_acc", acc, batch_size=len(y), on_epoch=True)  # Log accuracy
        self.log("val_f1", f1, batch_size=len(y), on_epoch=True)  # Log f1
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch)
        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds, y)  # Compute accuracy
        f1 = self.f1(preds, y)  # Compute f1
        self.log("test_acc", acc, batch_size=len(y), on_epoch=True)  # Log accuracy
        self.log("test_f1", f1, batch_size=len(y), on_epoch=True)  # Log f1
        return loss
    
    def predict_step(self, batch, batch_idx):
        _, logits, labels = self.step(batch)
        preds = torch.argmax(logits, dim=1)
        return labels, preds
    
    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer