import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from .data_utils import insert_markers
   
class GPTHandler:
    '''
    Base class for handling GPT-3 API calls.
    '''
    def __init__(self, lm_name='gpt-3.5-turbo-16k', args={'max_tokens': 500}):
        '''
        Initialize GPTHandler.
        Args:
        - lm_name (str): Name of the language model to use.
        - args (dict): Additional arguments for the model or API.
        '''
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_url = 'https://api.openai.com/v1/chat/completions'
        if not self.api_key:
            raise ValueError("API key must be set in environment variables.")
        self.lm_name = lm_name
        self.args = args
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _call_openai_api(self, data, max_retries=3, wait_time=10):
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=data)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.HTTPError as e:
                print(f"HTTP error: {e}, retrying...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    print("Max retries reached, skipping...")
                    return None

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
            prompt = f"Read the following text and answer the question below.\n{text}\n\nQ: Who is '{pronoun}' referring to?\nChoices: 1 {a}, 2 {b}, 3 neither.\nYour answer should be the number of the correct choice, i.e., 1, 2, or 3.\nAnswer:"
            
            # append to list
            prompts.append(prompt)
        
        return prompts
    
    def classify_coref_zs(self, data_path='data/gap/gap-test.tsv', num_samples=10, output_path=None):
        """
        Zero-shot coreference resolution on GAP dataset.

        Args:
        - data_path (str): Path to the data file.
        - num_samples (int): Number of samples to generate.
        - output_path (str): Path to save the generated results.

        Returns:
        - List[str]: List of responses.
        """
        responses = []
        prompts = self.create_coref_prompts_gap(data_path, num_samples)
        
        for prompt in tqdm(prompts):
            data = {
                "model": self.lm_name,
                "messages": [{"role": "system", "content": "You are a helpful assistant."},
                             {"role": "user", "content": prompt}],
                **self.args
            }
            out = self._call_openai_api(data)
            if out:
                responses.append(out)
        
        if output_path:
            with open(output_path, 'w') as f:
                for response in responses:
                    f.write(response)
                    f.write('\n')
                    
        return responses
    
    def check_grammar(self, text):
        '''
        Check & correct modified sentences for grammar using GPT-3.
        Args:
        - text (str): Text to check.
        Returns:
        - str: Corrected text.
        '''
        prompt = f"Read the following text and correct any grammatical errors. The text contains some special tokens beginning with the special characters '§', '¶' and ^. Leave them as is (including the special chars), do not modify them.\nText:{text}\n\nCorrected text:"
        
        data = {
            "model": self.lm_name,
            "messages": [{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}],
            **self.args  
        }
        
        out = self._call_openai_api(data)
        
        return out
        
