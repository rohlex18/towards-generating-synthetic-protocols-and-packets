import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MarkovModel2:
    def __init__(self, reg=0):
        self.reg = reg
        return None
    
    def train(self, FieldSequences):
        self.sequences = [fields + ['<END>'] for fields in FieldSequences.train_seq]
        self.init_dist = self.compute_initial_distribution()
        self.matrix, self.state_to_idx, self.unique_states = self.transition_matrix_categorical()

    def compute_initial_distribution(self):
        """ Compute the empirical distribution of initial states. """
        first_second_states = ['-'.join(seq[0:2]) for seq in self.sequences if seq[0] != "<END>"]
        unique_states, counts = np.unique(first_second_states, return_counts=True)
        probabilities = counts / counts.sum()
        return dict(zip(unique_states, probabilities))

    def transition_matrix_categorical(self):
        """ Build transition matrix from categorical sequences. """
        unique_states = sorted(set(state for seq in self.sequences for state in seq), reverse=True)
        state_to_idx = {state: i for i, state in enumerate(unique_states)}
        num_states = len(unique_states)

        # Initialize transition matrix
        matrix = np.zeros((num_states, num_states, num_states)) + self.reg

        # Count transitions
        for sequence in self.sequences:
            for i in range(len(sequence) - 2):
                from_idx = state_to_idx[sequence[i]]
                and_idx = state_to_idx[sequence[i + 1]]
                to_idx = state_to_idx[sequence[i + 2]]
                matrix[from_idx, and_idx, to_idx] += 1

        # Normalize rows to get probabilities
        row_sums = matrix.sum(axis=2, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)  # Avoid division by zero
        self.matrix=matrix
        return matrix, state_to_idx, unique_states
    
    def generate(self, N=5000):
        """ Generate N new sequences using learned transition probabilities until <END> is reached. """
        
        # Convert initial distribution to index-based
        init_probs = np.array(list(self.init_dist.values()))
        generated_sequences = []
        for _ in range(N):
            # Sample first state (excluding <END>)
            current_state = np.random.choice(np.array(list(self.init_dist.keys())), p=init_probs).split('-')
            sequence = current_state
            # Generate until <END> is reached
            while current_state[1] != "<END>":
                current_idx = self.state_to_idx[current_state[0]]
                and_idx = self.state_to_idx[current_state[1]]
                next_probs = self.matrix[current_idx, and_idx]
                if next_probs.sum() == 0:
                    break  # Stop if no transitions are possible
                next_state = np.random.choice(self.unique_states, p=next_probs)
                sequence.append(next_state)
                if next_state == "<END>":
                    break
                current_state = [current_state[1], next_state]

            generated_sequences.append(sequence)

        self.generated_sequences = generated_sequences
        return generated_sequences
    
    def gen_to_csv(self):
        rows = []
        for i,seq in enumerate(self.generated_sequences):
            if '<END>' in seq:
                rows.append(['MarkovGenerated2', f'ID:{i}'] + seq[:seq.index('<END>')])
            else:
                rows.append(['MarkovGenerated2', f'ID:{i}'] + seq)
        df = pd.DataFrame(rows)
        df.to_csv("CSVs/Markov2GeneratedHeaders.csv")
    
    def show_transitions(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
        ax.set_yticks(range(len(self.unique_states)))
        ax.set_yticklabels(self.state_to_idx.keys())
        ax.set_xticks(range(len(self.unique_states)))
        ax.set_xticklabels(list(self.state_to_idx.keys()), rotation=90)
        ax.set_ylabel('State N')
        ax.set_xlabel('State N+1')
        ax.imshow(self.matrix)

        plt.savefig(f'Plots/ScapyFieldTransitionMatrix.pdf', bbox_inches='tight',pad_inches = 0, format='pdf',backend='pdf', dpi=300)

        plt.show()