import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MarkovModel:
    def __init__(self, reg=0):
        self.reg=reg
        return None
    
    def train(self, FieldSequences):
        '''
        scapy_data = pd.read_csv(file)
        sequences = []
        for _, row in enumerate(scapy_data.iterrows()):
            fields = []
            try: 
                i = 3
                while pd.notna(row[1].iloc[i]) and i < len(row[1])-1:
                    field = row[1].iloc[i].split('(')[0]
                    fields.append(field)
                    i += 1
                sequences.append(fields + ['<END>'])
            except Exception as e:
                print(e, row)
        self.sequences = sequences
        '''
        self.sequences = [fields + ['<END>'] for fields in FieldSequences.train_seq]
        self.matrix, self.state_to_idx, self.unique_states = self.transition_matrix_categorical()
        self.init_dist = self.compute_initial_distribution()


    def compute_initial_distribution(self):
        """ Compute the empirical distribution of initial states. """
        #first_states = [seq[0] for seq in self.sequences if seq[0] != "<END>"]
        #unique_states, counts = np.unique(first_states, return_counts=True)
        #counts += self.reg
        #probabilities = counts / counts.sum()
        first_states = [seq[0] for seq in self.sequences if seq[0] != "<END>"]
        counts = np.zeros(len(self.unique_states)) + self.reg
        counts[self.state_to_idx['<END>']] = 0
        for s in first_states:
            counts[self.state_to_idx[s]] += 1
        probabilities = counts / counts.sum()
        return dict(zip(self.unique_states, probabilities))

    def transition_matrix_categorical(self):
        """ Build transition matrix from categorical sequences. """
        unique_states = sorted(set(state for seq in self.sequences for state in seq), reverse=True)
        state_to_idx = {state: i for i, state in enumerate(unique_states)}
        num_states = len(unique_states)

        # Initialize transition matrix
        matrix = np.zeros((num_states, num_states)) + self.reg

        # Count transitions
        for sequence in self.sequences:
            for i in range(len(sequence) - 1):
                from_idx = state_to_idx[sequence[i]]
                to_idx = state_to_idx[sequence[i + 1]]
                matrix[from_idx, to_idx] += 1

        # Normalize rows to get probabilities
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)  # Avoid division by zero
        self.matrix=matrix
        return matrix, state_to_idx, unique_states
    
    def generate(self, N=5000):
        """ Generate N new sequences using learned transition probabilities until <END> is reached. """
        
        # Convert initial distribution to index-based
        init_probs = np.array([self.init_dist.get(state, 0) for state in self.unique_states])
        
        generated_sequences = []
        for _ in range(N):
            # Sample first state (excluding <END>)
            current_state = np.random.choice(self.unique_states, p=init_probs)
            sequence = [current_state]
            
            # Generate until <END> is reached
            while current_state != "<END>":
                current_idx = self.state_to_idx[current_state]
                next_probs = self.matrix[current_idx]
                if next_probs.sum() == 0:
                    break  # Stop if no transitions are possible
                next_state = np.random.choice(self.unique_states, p=next_probs)
                sequence.append(next_state)
                if next_state == "<END>":
                    break
                current_state = next_state

            generated_sequences.append(sequence)

        self.generated_sequences = generated_sequences
        return generated_sequences
    
    def gen_to_csv(self):
        rows = []
        for i,seq in enumerate(self.generated_sequences):
            if '<END>' in seq:
                rows.append(['MarkovGenerated', f'ID:{i}'] + seq[:seq.index('<END>')])
            else:
                rows.append(['MarkovGenerated', f'ID:{i}'] + seq)
        df = pd.DataFrame(rows)
        df.to_csv("CSVs/MarkovGeneratedHeaders.csv")
    
    def show_transitions(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
        ax.set_yticks(range(len(self.unique_states)))
        ax.set_yticklabels(list(self.state_to_idx.keys())[:-1]+['\\textless END \\textgreater'])
        ax.set_xticks(range(len(self.unique_states)))
        ax.set_xticklabels(list(self.state_to_idx.keys())[:-1]+['\\textless END \\textgreater'], rotation=90)
        ax.set_ylabel('State N',size=24)
        ax.set_xlabel('State N+1',size=24)
        ax.imshow(self.matrix)

        plt.savefig(f'Plots/ScapyFieldTransitionMatrix.pdf', bbox_inches='tight',pad_inches = 0, format='pdf',backend='pdf', dpi=300)
        plt.show()