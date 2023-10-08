import os
import time
import requests
from dotenv import load_dotenv


def classify_coref_zs(data_path='data/gap/gap-test.tsv', num_samples=10, lm_name='gpt-3.5-turbo-16k', args={'max_tokens': 500}):
    """
    Summarize path reports using zero-shot generation.

    Args:
    - data_path (str): Path to the data file.
    - num_samples (int): Number of samples to generate.
    - reports_sum_dir (str): Directory to save the summary files.
    - lm_name (str): Name of the language model to use.
    - args (dict): Additional arguments for the model or API.

    Returns:
    - DataFrame: Pandas DataFrame containing the classification results.
    """
    
    # call openai
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    API_URL = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    responses = []  # list to store responses
    
    # generate prompts
    prompts = create_coref_prompts_gap(data_path, num_samples)
    
    # Loop through all prompts and generate responses
    for prompt in prompts:
        
        data = {
            "model": lm_name,
            "messages": [{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}],
            **args  # Add any additional arguments
        }
        
        # Retry mechanism
        max_retries = 3
        wait_time = 10  # in seconds
        for attempt in range(max_retries):
            try:
                # call openai text generation api
                response = requests.post(API_URL, headers=headers, json=data)
                response.raise_for_status()
                out = response.json()["choices"][0]["message"]["content"]
                # write cleaned data to file
                # with open(dest_file, 'w') as f:
                    # f.write(out)
                responses.append(out)
                
            except requests.HTTPError as e:
                print(f"HTTP error: {e}, retrying...")
                if attempt < max_retries - 1:  # i.e. if it's not the last attempt
                    time.sleep(wait_time)  # wait for a bit before retrying
                else:
                    print("max retries reached, skipping...")
                    
    return responses

def create_coref_prompts_gap(data_path='data/gap/gap-test.tsv', num_samples=10):
    '''
    Create prompts for coreference resolution using GAP dataset.
    Args:
    - data_path (str): Path to GAP dataset.
    '''
    with open(data_path, 'r') as f:
        lines = f.readlines()
    # take only the first num_samples
    lines = lines[1:num_samples+1]
    # initialize list to store prompts
    prompts = []
    for line in lines:
        # get elements
        id, text, pronoun, pronoun_offset, a, a_offset, a_coref, b, b_offset, b_coref, url = line.split('\t')
        
        # create prompt
        prompt = f"Read the following text and answer the question below.\n{text}\n\nQ: Who is '{pronoun}' referring to?\nChoices: {a}, {b}, neither.\nYour answer should EXACTLY match one of the choices, with no additional words."
        
        # append to list
        prompts.append(prompt)
    
    return prompts