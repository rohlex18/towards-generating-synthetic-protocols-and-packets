from llama_cpp import Llama
import random
import torch
import pandas as pd


class LlamaModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'{self.device} enabled')

    def train(self, R):
        self.sequences =  [fields + ['<END>'] for fields in R.train_seq]
        self.llm = llm = Llama(
                        model_path="codellama-7b-instruct.Q5_K_M.gguf",
                        n_ctx=2048,
                        n_threads=28,  # match your CPU
                        n_batch=512,
                        n_gpu_layers=35
                        )
        self.unique_fields = R.unique_fields   

    def generate(self, n=5000, max_len=40, temperature=1.5):
        self.generated_sequences = []
        for i in range(n):
            if i%100 == 0:
                print(f'{i}/{n}')
            pn = 5
            injects = random.sample(self.sequences, pn)
            prompt =   """### Task: Generate a list of field values in a realistic sequence.
                        Each example ends with an <END> token.
                        Do not explain or add anything extra.\n"""
            for inject in injects:
                prompt += f'Example: ' + ' '.join(inject) + '\n'
            prompt += 'New Example:'
            print(prompt)
            response = self.llm(prompt, max_tokens=50)
            print(f'{response["choices"][0]["text"]=}')
            seq = response["choices"][0]["text"].split('<END>')[0].strip()
            seq = seq.split(' ') + ['<END>']
            seq = [tok for tok in seq if tok in self.unique_fields] + ['<END>']
            print(f'{seq=}')
            self.generated_sequences.append(seq)
    
    def gen_to_csv(self):
        rows = []
        for i,seq in enumerate(self.generated_sequences):
            if '<END>' in seq:
                rows.append(['LlamaGenerated', f'ID:{i}'] + seq[:seq.index('<END>')])
            else:
                rows.append(['LlamaGenerated', f'ID:{i}'] + seq)
        df = pd.DataFrame(rows)
        df.to_csv("CSVs/LlamaGeneratedHeaders.csv")


