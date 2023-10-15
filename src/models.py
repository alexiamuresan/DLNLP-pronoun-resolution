import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
import torchmetrics
from fastcoref import FCoref, LingMessCoref
from .data_utils import *


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
    
class FCorefClassifier:
    def __init__(self, model_name='lmcoref', data_path='data/gap/gap-test.tsv', num_samples=None):
        """
        Initialize the FCorefClassifier.

        Parameters:
        - model_name (str): Name of the coreference resolution model ('fcoref' or 'lingmesscoref').
        - data_path (str): Path to the GAP dataset.
        - num_samples (int or None): Number of samples to load from the GAP dataset.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_name)
        self.data_path = data_path
        self.num_samples = num_samples
        self.samples = self._load_samples()
        
    def _load_model(self, model_name):
        """
        Load the specified coreference resolution model.

        Parameters:
        - model_name (str): Name of the model to load.

        Returns:
        - model: Loaded coreference resolution model.
        """
        # choose the model based on the input model name
        if model_name == 'fcoref':
            return FCoref(device=self.device)
        elif model_name == 'lmcoref':
            return LingMessCoref(device=self.device)
        else:
            raise ValueError("Invalid model_name. Choose 'fcoref' or 'lingmesscoref'.")
    
    def _load_samples(self):
        """
        Load a random sample of data from the GAP dataset.

        Returns:
        - DataFrame: Sampled data from the GAP dataset.
        """
        df_gap = pd.read_csv(self.data_path, sep='\t')
        if self.num_samples:
            df_gap = df_gap.sample(self.num_samples)
        return df_gap
    
    def pred_cr_clusters(self, output_path=None, verbose=False):
        """
        Predict coreference clusters for the sample texts.
        Parameters:
        - output_path (str or None): Path to save the predicted clusters.
        - verbose (bool): Whether to print verbose output.
        Returns:
        - List: Predicted coreference clusters for each text.
        """
        texts = self.samples['Text'].tolist()
        cluster_preds = self.model.predict(texts, output_file=output_path)
        
        # print verbose output
        if verbose:
            for text, cluster_pred in zip(texts, cluster_preds):
                print(f"text: {text}\nclusters: {cluster_pred.get_clusters(as_strings=True)}\n")
        
        return cluster_preds
    
    def pred_cr_labels(self, verbose=False):
        """
        Predict coreference labels (1 for A-coref, 2 for B-coref, 0 for NEITHER) 
        based on the predicted clusters.
        Parameters:
        - verbose (bool): Whether to print verbose output.
        Returns:
        - List: Predicted labels for each text.
        """
        # predict clusters
        texts = self.samples['Text'].tolist()
        cluster_preds = self.model.predict(texts)
        labels, preds = [], []
        
        # iterate through each text and its corresponding predicted clusters
        print(f'predicting labels for {len(texts)} samples using {self.model.__class__.__name__}...')
        for idx, (text, cluster_pred) in tqdm(enumerate(zip(texts, cluster_preds))):
            sample = self.samples.iloc[idx]
            label = 1 if sample['A-coref'] else 2 if sample['B-coref'] else 0
            labels.append(label)
            
            # extract spans
            p_span = get_span_gap(sample['Pronoun'], sample['Pronoun-offset'])
            a_span = get_span_gap(sample['A'], sample['A-offset'])
            b_span = get_span_gap(sample['B'], sample['B-offset'])
            
            # get clusters as spans
            clusters = cluster_pred.get_clusters(as_strings=False)
            
            # check if pronoun and A/B are in the same cluster
            pronoun_cluster = [cluster for cluster in clusters if p_span in cluster]
            a_cluster = [cluster for cluster in clusters if a_span in cluster]
            b_cluster = [cluster for cluster in clusters if b_span in cluster]
            
            # determine predicted label
            if pronoun_cluster and a_cluster and pronoun_cluster[0] == a_cluster[0]:
                pred = 1
            elif pronoun_cluster and b_cluster and pronoun_cluster[0] == b_cluster[0]:
                pred = 2
            else:
                pred = 0
            
            preds.append(pred)
            
            # print verbose output
            if verbose:
                print(f"text: {text}\nP: {sample['Pronoun']}, A: {sample['A']}, B: {sample['B']}, label: {label}, pred: {pred}\n")
        print('done.')
        
        return labels, preds



