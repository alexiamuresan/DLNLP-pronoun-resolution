import pandas as pd
from pathlib import Path
from collections import Counter
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer

class GAPDataset(Dataset):
    def __init__(self, df, tokenizer='roberta-large', max_len=256):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # process the row to return the necessary information as a dictionary.
        
        # sample = {
        #     'text': row['Text'],
        #     'pronoun': row['Pronoun'],
        #     'pronoun_offset': row['Pronoun-offset'],
        #     'a': row['A'],
        #     'a_offset': row['A-offset'],
        #     'a_coref': 1 if row['A-coref'] else 0,
        #     'b': row['B'],
        #     'b_offset': row['B-offset'],
        #     'b_coref': 1 if row['B-coref'] else 0,
        #     'url': row['URL'],
        #     'label': 0 if row['A-coref'] else 1 if row['B-coref'] else 2
        # }
        
        text, pronoun, pronoun_offset, a, a_offset, b, b_offset = row['Text'], row['Pronoun'], row['Pronoun-offset'], row['A'], row['A-offset'], row['B'], row['B-offset']
        a_coref = 1 if row['A-coref'] else 0
        b_coref = 1 if row['B-coref'] else 0
        url = row['URL']
        label = 0 if row['A-coref'] else 1 if row['B-coref'] else 2
        
        modified_text = text[:pronoun_offset] + '[PRONOUN] ' + pronoun + ' [PRONOUN]' + text[pronoun_offset + len(pronoun):a_offset] + '[ENTITY_A] ' + a + ' [ENTITY_A]' + text[a_offset + len(a):b_offset] + '[ENTITY_B] ' + b + ' [ENTITY_B]' + text[b_offset + len(b):]
        
        # tokenize
        inputs = self.tokenizer(modified_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
        
        # convert to tensors and return
        sample = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'a_coref': torch.tensor(a_coref),
            'b_coref': torch.tensor(b_coref),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        return sample

class GAPDataModule(LightningDataModule):
    def __init__(self, root_dir='data/gap/', tokenizer='roberta-large', max_len=256, bsz=32):
        super().__init__()
        self.train_path = Path(root_dir) / 'gap-development.tsv'
        self.val_path = Path(root_dir) / 'gap-validation.tsv'
        self.test_path = Path(root_dir) / 'gap-test.tsv'
        self.bsz = bsz

        # Load data from the three separate files.
        self.train_df = pd.read_csv(self.train_path, sep='\t')
        self.val_df = pd.read_csv(self.val_path, sep='\t')
        self.test_df = pd.read_csv(self.test_path, sep='\t')

        # Assign train/val datasets for use in dataloaders.
        self.train_dataset = GAPDataset(self.train_df, tokenizer, max_len)
        self.val_dataset = GAPDataset(self.val_df, tokenizer, max_len)
        self.test_dataset = GAPDataset(self.test_df, tokenizer, max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.bsz, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.bsz, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.bsz, num_workers=12)


def get_value_counts(data_module):
    def get_label_counts(dataset):
        return Counter(sample['label'] for sample in dataset)

    datasets = {'train': data_module.train_dataset, 
                'val': data_module.val_dataset, 
                'test': data_module.test_dataset}

    for split, dataset in datasets.items():
        label_counts = get_label_counts(dataset)
        print(split)
        for label, count in label_counts.items():
            print(f"{label}: {count}")
            
def get_span_gap(entity, start_offset):
    """
    Get the span of the entity in the GAP dataset.

    Parameters:
    - entity (str): The entity string.
    - start_offset (int): The character offset of the entity in the text.

    Returns:
    - tuple: (start_offset, end_offset)
    """
    return (start_offset, start_offset + len(entity))

def convert_gap_to_jsonlines(gap_data_path, output_jsonl_path, num_samples=None):
    """
    Convert GAP dataset to jsonlines format suitable for training with FCoref.

    Parameters:
    - gap_data_path (str): Path to the GAP dataset in .tsv format.
    - output_jsonl_path (str): Path to save the output .jsonlines file.
    - num_samples (int, optional): Number of samples to convert. If None, convert all.

    Outputs:
    - Writes a .jsonlines file at output_jsonl_path.
    """
    # load gap data
    df_gap = pd.read_csv(gap_data_path, sep='\t')
    
    # optionally, sample a subset of the data
    if num_samples is not None:
        df_gap = df_gap.sample(num_samples)
    
    # open output file
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        # iterate through each row in the dataframe
        for _, row in df_gap.iterrows():
            # extract relevant information
            text = row['Text']
            a_coref = row['A-coref']
            b_coref = row['B-coref']
            
            # get spans using the utility function
            a_span = get_span_gap(row['A'], row['A-offset'])
            b_span = get_span_gap(row['B'], row['B-offset'])
            pronoun_span = get_span_gap(row['Pronoun'], row['Pronoun-offset'])
            
            # create clusters and cluster strings
            clusters = []
            clusters_strings = []
            if a_coref:
                clusters = [[a_span, pronoun_span], [b_span]]
                clusters_strings = [[row['A'], row['Pronoun']], [row['B']]]
            elif b_coref:
                clusters = [[b_span, pronoun_span], [a_span]]
                clusters_strings = [[row['B'], row['Pronoun']], [row['A']]]
            else:
                clusters = [[a_span], [b_span], [pronoun_span]]
                clusters_strings = [[row['A']], [row['B']], [row['Pronoun']]]
            
            # create json object
            json_obj = {
                "text": text,
                "clusters": clusters,
                "clusters_strings": clusters_strings
            }
            
            # write to file
            f_out.write(json.dumps(json_obj) + '\n')