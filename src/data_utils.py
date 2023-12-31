import pandas as pd
from collections import Counter
import re
from tqdm import tqdm
from .zs_utils import GPTHandler


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

def insert_markers(sample):
    """
    Insert unique markers before the pronoun, entity A, and entity B in the text.
    
    Parameters:
    - sample (dict): A sample from the GAP dataset.
    
    Returns:
    - str: The modified text with markers inserted.
    """
    # extract the text and offsets
    text, p_offset, a_offset, b_offset = sample['Text'], sample['Pronoun-offset'], sample['A-offset'], sample['B-offset']
    
    # define unique markers
    p_marker = "§"
    a_marker = "¶"
    b_marker = "^"
    
    # create a list of (offset, marker) pairs and sort them by offset
    markers = [(p_offset, p_marker), (a_offset, a_marker), (b_offset, b_marker)]
    markers.sort(key=lambda x: x[0])
    
    # convert the text to a list
    text_list = list(text)
    
    # insert markers with additional offset according to their order
    for i, (offset, marker) in enumerate(markers):
        text_list.insert(offset + i, marker)
    
    # convert the modified text list back to str
    modified_text = ''.join(text_list)
    
    return modified_text

def replace_pronoun_with_neutral(text):
    '''
    Replace gendered pronouns w/ gender-neutral in the text.
    Args:
    - text (str): The text to modify.
    '''
    pronoun_mapping = {
        "he": "they",
        "she": "they",
        "his": "their",
        "her": "their",
        "him": "them",
        "himself": "themselves",
        "herself": "themselves"
        # expand this list as needed
    }
    
    # find the pronoun immediately following '@' and replace it
    for gendered, neutral in pronoun_mapping.items():
        pattern = r'§' + re.escape(gendered)
        replacement = '§' + neutral
        # replace only the first occurrence
        text, num_replaced = re.subn(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        # if a replacement is made, return the modified text and the neutral pronoun
        if num_replaced == 1:
            return text, neutral
        
    # if no pronoun is found, return None
    return text, None       

def remove_markers(text):
    """
    Remove markers from the text and compute the final offsets for pronoun, A, and B.
    
    Parameters:
    - text (str): The modified text containing markers before pronoun, A, and B.
    
    Returns:
    - str: The text with markers removed.
    - dict: The final offsets for pronoun, A, and B.
    
    Raises:
    - ValueError: If one or more markers are not found in the text.
    """
    # find the positions of the markers
    p_marker_pos = text.find('§')
    a_marker_pos = text.find('¶')
    b_marker_pos = text.find('^')
    
    # ensure all markers are found
    if a_marker_pos == -1 or b_marker_pos == -1 or p_marker_pos == -1:
        raise ValueError("One or more markers not found in text.")
    
    # create a list of (marker_position, original_offset_name) tuples
    marker_positions = [(a_marker_pos, 'A-offset'), (b_marker_pos, 'B-offset'), (p_marker_pos, 'Pronoun-offset')]
    
    # sort the marker positions so we remove them in the right order
    marker_positions.sort(key=lambda x: x[0])
    
    # convert string to list for easier character removal
    text_list = list(text)
    
    # initialize new offsets dictionary
    new_offsets = {}
    
    # remove markers and compute new offsets
    for i, (marker_pos, offset_name) in enumerate(marker_positions):
        # compute new offset: position of the marker minus the index
        new_offsets[offset_name] = marker_pos - i
        
        # remove the marker from the text list
        text_list.pop(marker_pos - i)
    
    # convert list back to string
    text = ''.join(text_list)
    
    return text, new_offsets

def create_gn_gap(data_path='data/gap/gap-test.tsv', output_path='data/gap/gap-test-gn.tsv'):
    '''
    Create a gender-neutral version of GAP by replacing gendered pronouns w/ gender-neutral.
    
    Args:
    - data_path (str): Path to GAP dataset.
    - output_path (str): Path to save the modified dataset.
    Returns:
    - pd.DataFrame: The modified GAP dataset.
    '''
    
    # read the data
    df_gap = pd.read_csv(data_path, sep='\t')
    df_gap_gn = df_gap.copy()
    
    # init GPThandler (for grammar)
    gpt = GPTHandler()
    
    # iterate thru rows and replace pronouns
    for i, row in tqdm(df_gap.iterrows()):
        # insert markers
        text = insert_markers(row)
        # replace pronouns
        text, new_p = replace_pronoun_with_neutral(text)
        # correct grammar (optional)
        # text = gpt.check_grammar(text)
        # remove markers & get new offsets
        text, offsets = remove_markers(text)
        # update the df
        df_gap_gn.loc[i, 'Text'] = text # update the text
        df_gap_gn.loc[i, 'Pronoun'] = new_p if new_p else row['Pronoun']  # update pronoun if it was replaced
        df_gap_gn.loc[i, 'Pronoun_old'] = row['Pronoun']    # save the old pronoun
        # update the offsets
        for offset_name, offset in offsets.items():
            df_gap_gn.loc[i, offset_name] = offset
            
    # save the modified df
    df_gap_gn.to_csv(output_path, sep='\t', index=False)
    
    return df_gap_gn

def check_offsets(modified_data_path, num_rows=None):
    """
    Check if the computed offsets in the modified dataset correctly point to the entities.

    Parameters:
    - modified_data_path (str): Path to the modified dataset in .csv format.
    - num_rows (int, optional): Number of rows to check. If None, check all rows.

    Outputs:
    - Prints mismatches if there are any.
    """
    # load the modified dataset
    df = pd.read_csv(modified_data_path, sep='\t')
    
    # optionally, check a subset of the data
    if num_rows is not None:
        df = df.sample(min(num_rows, len(df)))
    
    # iterate through each row in the dataframe
    for _, row in df.iterrows():
        # check each entity and its offset
        for entity_type in ['A', 'B', 'Pronoun']:
            entity = row[entity_type]
            offset = row[f"{entity_type}-offset"]
            
            # fetch the entity using the offset and compare with the actual entity
            fetched_entity = row['Text'][offset:offset+len(entity)]
            if fetched_entity != entity:
                print(f"Mismatch in row {row['ID']} for entity {entity_type}:")
                print(f"  Expected: {entity}, Fetched: {fetched_entity}\n")