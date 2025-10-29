import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class GeneticModel2:
    def __init__(self):
        return None
    
    def train(self, FieldSequences):
        self.sequences = [fields + ['<END>'] for fields in FieldSequences.train_seq]
        self.unique_fields = FieldSequences.unique_fields
    
    def generate(self, N=5000, reg=10):
        """ Generate N new sequences """
        
        generated_sequences = []
        for _ in range(N):
            # Sample first state (excluding <END>)
            random_sequence = random.choice(self.sequences)
            lst = random_sequence[:-1]  # make a copy to avoid modifying original
            if reg>20:
                reg = len(lst)
            for _ in range(reg):
                old_item = random.choice(lst)
                new_item = random.choice(self.unique_fields)
                lst = [new_item if item == old_item else item for item in lst]
            
            sequence = lst + ['<END>']
            generated_sequences.append(sequence)

        self.generated_sequences = generated_sequences
        return generated_sequences
    
    def gen_to_csv(self):
        rows = []
        for i,seq in enumerate(self.generated_sequences):
            if '<END>' in seq:
                rows.append(['Genetic2Generated', f'ID:{i}'] + seq[:seq.index('<END>')])
            else:
                rows.append(['Genetic2Generated', f'ID:{i}'] + seq)
        df = pd.DataFrame(rows)
        df.to_csv("CSVs/Genetic2GeneratedHeaders.csv")