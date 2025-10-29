import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class GeneticModel:
    def __init__(self):
        return None
    
    def train(self, FieldSequences):
        self.sequences = [fields + ['<END>'] for fields in FieldSequences.train_seq]
    
    def generate(self, N=5000, reg=10):
        """ Generate N new sequences """
        
        generated_sequences = []
        for _ in range(N):
            # Sample first state (excluding <END>)
            random_sequence = random.choice(self.sequences)
            lst = random_sequence[:-1]  # make a copy to avoid modifying original
            if len(lst)>=2 and reg<20:
                for _ in range(reg):
                    i, j = random.sample(range(len(lst)), 2)
                    lst[i], lst[j] = lst[j], lst[i]
                sequence = lst + ['<END>']
            elif reg >= 20:
                sequence = random.sample(random_sequence[:-1], len(random_sequence)-1) + ['<END>']
            else:
                sequence = random_sequence
            generated_sequences.append(sequence)

        self.generated_sequences = generated_sequences
        return generated_sequences
    
    def gen_to_csv(self):
        rows = []
        for i,seq in enumerate(self.generated_sequences):
            if '<END>' in seq:
                rows.append(['GeneticGenerated', f'ID:{i}'] + seq[:seq.index('<END>')])
            else:
                rows.append(['GeneticGenerated', f'ID:{i}'] + seq)
        df = pd.DataFrame(rows)
        df.to_csv("CSVs/GeneticGeneratedHeaders.csv")