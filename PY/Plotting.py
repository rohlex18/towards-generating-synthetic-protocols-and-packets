import pandas as pd 
import numpy as np
import math
from collections import Counter
from adjustText import adjust_text
import matplotlib.cm as cm
import random
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': '\\usepackage[dvips]{graphicx}\\usepackage{xfrac}\\usepackage{amssymb}\\usepackage{lmodern}'
})

def calculate_entropy(strings):
    # Count the frequency of each string
    frequency = Counter(strings)
    
    # Total number of strings
    total_strings = len(strings)
    
    # Calculate entropy
    entropy = 0
    for count in frequency.values():
        probability = count / total_strings
        entropy -= probability * math.log2(probability)
    
    return entropy

def logfunc(n, k):
    k-=1
    if k==0:
        return 0
    else:
        return -1 * (k*math.log2(1/n)/n + ((n-k)/n) * math.log2((n-k)/n))

class FieldSequencesPlotter:
    def __init__(self, data_file, val_set = False):
        self.val = val_set
        data_file = 'CSVs/FinalExperiment/'+data_file
        data = pd.read_csv(data_file, low_memory=False)
        self.name = data_file.split('/')[-1].split('.')[0]
        self.unique_fields = []
        max = 5
        self.all_fields = []
        self.all_pros = []
        self.count_dict = {}
        for i, row in enumerate(data.iterrows()):
            pro = row[1].iloc[1].split('.')[0].split('/')[-1] + '/' + row[1].iloc[2]
            fields = []
            i = 3
            try: 
                while pd.notna(row[1].iloc[i]) and i < len(row[1])-1:
                    field = row[1].iloc[i].split('(')[0]
                    self.count_dict[field] = self.count_dict.get(field, 0) + 1
                    fields.append(field)
                    if field not in self.unique_fields:
                        self.unique_fields.append(field)
                    i +=1
            except Exception as e:
                print(e, row)
            self.all_fields.append(fields)
            self.all_pros.append(pro)
        self.unique_field_sequences = list(map(list, {tuple(lst) for lst in self.all_fields}))
        if val_set:
            val_ind, train_ind = generate_indices(self.unique_field_sequences, val_set)
            self.train_seq = split_list_by_indices(self.unique_field_sequences, train_ind)
            self.val_seq = split_list_by_indices(self.unique_field_sequences, val_ind)
        else:
            self.train_seq = self.unique_field_sequences

    def plot_entropy_on_numfields(self, show_texts=False):
        """ plot the data found in a csv 
        csv format: index, prototype, name, field 1, field 2
        e.g. 1,scapy/scapy/arch/bpf/pfroute.py,pfmsghdr,"Field('rtm_msglen', 0, fmt='=H')","ByteField('rtm_ver
        """
        cmap = plt.get_cmap('tab10').colors # cm.viridis  # Choose a colormap (you can change it to any other colormap)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
        X = 'Number of Fields'
        Y = 'Packet Format Entropy'      
        texts = []
        points = []

        for fields, pro in zip(self.all_fields, self.all_pros):
            pro_x = len(fields)
            pro_y = calculate_entropy(fields)
            pro_c = len(set(fields))#/max

            try:
                a = ax.scatter(pro_x, pro_y, s=40, alpha=0.2,color=cmap[pro_c-1])#, cmap=cmap)#, vmin=1, vmax=10)
            except Exception as e:
                a = ax.scatter(pro_x, pro_y, s=40, alpha=0.2,c='black')

            if show_texts:    
                #if pro_x > 14 or pro_y > 2.8:
                    #ax.text(pro_x, pro_y, pro, fontdict={'size':8, 'color':'black', 'alpha':1})
                 #   texts.append(plt.text(pro_x, pro_y, pro, fontdict={'size':8, 'color':'black', 'alpha':1}))
                for k, v in show_texts.items():
                    if pro.split('/')[-1] == v[0]:
                        texts.append(plt.text(pro_x, pro_y, k, fontdict={'size':12, 'color':'black', 'alpha':1}))
                        points.append((pro_x, pro_y))

        fields = ['XShortEnumField', 'XShortEnumField', 'FieldLenField', 'FieldLenField', 'ShortEnumField', 'SourceMACField', 'SourceIPField', 'MACField', 'IPField']
        pro_x = len(fields)
        pro_y = calculate_entropy(fields)
        texts.append(plt.text(pro_x, pro_y, "ARP", fontdict={'size':12, 'color':'black', 'alpha':1}))
        points.append((pro_x, pro_y))

       
        adjust_text(
            texts,
            autoalign=False,
            only_move={'points': 'none', 'text': 'xy'},  # ⬅️ Only text moves, in x and y
        )

        renderer = fig.canvas.get_renderer()

        for (x0, y0), txt in zip(points, texts):
            # Get bounding box of text in display coords
            bbox = txt.get_window_extent(renderer=renderer)
            # Convert bbox width from pixels to data coords
            inv = ax.transData.inverted()
            bbox_data_coords = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
            width_data = bbox_data_coords[1][0] - bbox_data_coords[0][0]

            x1, y1 = txt.get_position()
            left_edge_x = x1 - width_data / 4  # move from center to left edge

            ax.plot([x0, left_edge_x], [y0, y1], color='gray', lw=0.5)



        ax.set_xlabel(X,size=16)
        ax.set_ylabel(Y,size=16)

        matplotlib.rcParams['xtick.bottom'] = True
        matplotlib.rcParams['ytick.left'] = True

        xs = np.linspace(2,30,300)

        for k in range(1,10):
            xs = np.linspace(k,60,300)
            plt.plot(xs, [logfunc(n, k) for n in xs], label = f"{k} class{'es' if k != 1 else ''}", c=cmap[k-1], alpha=0.3, linestyle='--')

        plt.xlim((0.9,30))
        plt.ylim((-0.1,3.5))
        plt.legend(title="Minimum entropy for:", ncol=2, fontsize=14,title_fontsize=16)
        plt.grid(True, which='major', axis='x', linestyle='--', color='gray', alpha=0.7)

        plt.savefig(f'Plots/{self.name}EntropyOnNumFields.pdf', bbox_inches='tight',pad_inches = 0, format='pdf',backend='pdf', dpi=300)

        plt.show()

    def plot_all_field_dist(self):
        """ plot the data found in a csv 
        csv format: index, prototype, name, field 1, field 2
        e.g. 1,scapy/scapy/arch/bpf/pfroute.py,pfmsghdr,"Field('rtm_msglen', 0, fmt='=H')","ByteField('rtm_ver
        """

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))

        counts = dict(sorted(self.count_dict.items(), key=lambda item: item[1], reverse=True))
        self.counts = counts
        plt.bar(x = counts.keys(), height = counts.values())
        plt.xticks(rotation=90)
        plt.ylabel('Count of Field',size=24)
        plt.xlabel('Field Scapy Type',size=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(f'Plots/{self.name}FieldScapyTypeHist.pdf', bbox_inches='tight',pad_inches = 0, format='pdf',backend='pdf', dpi=300)
        plt.show()


    def plot_all_sequence_lengths(self):
        """
        Plot histogram of field sequence lengths and overlay a Poisson distribution.
        """
        import matplotlib.pyplot as plt
        from scipy.stats import poisson, chisquare
        import numpy as np
        fig, ax = plt.subplots(figsize=(12, 8))

        # Sequence lengths
        counts = [len(seq) for seq in self.unique_field_sequences]# if len(seq)<15]

        # Histogram data
        max_len = max(counts)
        value_range = range(1, max_len + 1)
        hist_counts = [counts.count(v) for v in value_range]

        # Plot histogram
        ax.bar(value_range, hist_counts)

        # Fit and plot Poisson distribution
        mu = np.mean(counts)
        poisson_pmf = [poisson.pmf(k, mu) * len(counts) for k in value_range]

        # Compute expected frequencies under Poisson(μ)
        expected_probs = poisson.pmf(value_range, mu)
        expected_freqs = expected_probs * sum(hist_counts)
        expected_freqs *= sum(hist_counts) / expected_freqs.sum()  # normalize to match total


        # Chi-squared test
        chi2_stat, p_value = chisquare(f_obs=hist_counts, f_exp=expected_freqs)
        #ax.plot(value_range, poisson_pmf, color='red', marker='o', label=f"Poisson PMF ($\mu$={mu:.2f}, p={p_value:.2f})")


        # Print result
        print(f"Chi² Statistic: {chi2_stat:.3f}, p-value: {p_value:.5f}")

        # Labels and formatting
        #plt.xticks(rotation=90)
        plt.ylabel('Count of Length', size=24)
        plt.xlabel('Field Sequence Length', size=24)
        #plt.title(rf"Sequence Length Histogram", size=18)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True, which='major', axis='y', linestyle='--', color='gray', alpha=0.5)
        # Save and show
        plt.savefig(f'Plots/{self.name}FieldScapyLengthHist.pdf', bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)
        plt.show()



    def self_statistics(self):
        self.diversity = len(self.unique_field_sequences)/len(self.all_fields)
        print('Diversity Score', len(self.unique_field_sequences), '/', len(self.all_fields))
        print('Example', self.all_fields[-1])
        '''
        count_dict = {}
        for fs in self.all_fields:
            key = ', '.join(fs)
            count_dict[key] = count_dict.get(key, 0) + 1
        sorted_dict = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))
        c=0
        for key, value in sorted_dict.items():
            print(key, value)
            c+=1
            if c>5:
                break
        '''
        self.entropies = [calculate_entropy(fs) for fs in self.all_fields]
        self.lengths = [len(fs) for fs in self.all_fields]
        self.unique_lengths = [len(fs) for fs in self.unique_field_sequences]
        self.len_c_dict = {}
        for l in range(max(self.unique_lengths)):
            self.len_c_dict[l] = self.unique_lengths.count(l)

    def comparative_statistics(self, Real):
        # see presentation diagram
        self.Gn = sum([1 if fs not in Real.unique_field_sequences else 0 for fs in self.unique_field_sequences])
        self.G = len(self.all_fields)
        if Real.val:
            self.VnG = sum([1 if fs in self.unique_field_sequences else 0 for fs in Real.val_seq])
            self.V = len(Real.val_seq)
            self.score = (self.VnG / self.V) + (self.Gn / self.G)
        self.T = len(Real.train_seq)
        self.TnG = sum([1 if fs in self.unique_field_sequences else 0 for fs in Real.train_seq])
        self.score2 = 10*(self.VnG / self.V) + (self.Gn / self.G) #(self.TnG / self.T) + (self.Gn / self.G)
        print(f'{self.TnG=}, {self.T=}, {self.Gn=}, {self.G=}, {self.score2=}')

    def hist_plot(self):
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(12, 8))

        X = 'Number of Fields'
        Y = 'Packet Format Entropy'

        # Prepare data lists
        pro_xs = []
        pro_ys = []

        for fields, pro in zip(self.all_fields, self.all_pros):
            pro_x = len(fields)
            pro_y = calculate_entropy(fields)
            pro_xs.append(pro_x)
            pro_ys.append(pro_y)

        # Convert to numpy arrays for convenience
        pro_xs = np.array(pro_xs)
        pro_ys = np.array(pro_ys)

        # Plot 2D histogram (hexbin)
        hb = ax.hexbin(pro_xs, pro_ys, gridsize=40, cmap='viridis', mincnt=1)

        # Add colorbar
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Counts')

        ax.set_xlabel(X)
        ax.set_ylabel(Y)
        #ax.set_title('2D Histogram of Number of Fields vs Header Field Entropy')

        plt.show()



def generate_indices(lst, percent):
    """Generates two sets of indices: 'in' (selected) and 'out' (remaining)."""
    split_size = int(len(lst) * percent)
    in_indices = set(random.sample(range(len(lst)), split_size))  # Selected indices
    out_indices = set(range(len(lst))) - in_indices  # Remaining indices
    return in_indices, out_indices

def split_list_by_indices(lst, indices):
    """Splits a list into two parts using predefined indices."""
    return [lst[i] for i in indices]  # Elements at selected indices


import matplotlib.pyplot as plt
from collections import Counter

def plot_protocol_distribution(dataset_loader):
    from collections import defaultdict

    protocol_label_counts = defaultdict(lambda: [0, 0])  # {protocol: [real_count, fake_count]}

    for _, labels, protocols in dataset_loader:
        for label, proto in zip(labels, protocols):
            protocol_label_counts[proto][label] += 1  # label 0 = real, 1 = fake

    protocols = list(protocol_label_counts.keys())
    real_counts = [protocol_label_counts[p][0] for p in protocols]
    fake_counts = [protocol_label_counts[p][1] for p in protocols]
    # Get default color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c['color'] for c in prop_cycle]
    plt.figure(figsize=(10, 5))
    plt.bar(protocols, real_counts, color=colors, label='Real')
    #plt.bar(protocols, fake_counts, bottom=real_counts, label='Fake', color='salmon')
    plt.xlabel('Protocol', size=24)
    plt.ylabel('Number of Samples', size=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, which='major', axis='y', linestyle='--', color='gray', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Plots/TrainProtoDistribution.png')
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_all_runs(all_losses, metric, save=None):
    # all_losses: list of lists, shape (num_runs, num_epochs)
    all_losses_arr = np.array(all_losses)  # shape (runs, epochs)

    mean_losses = np.mean(all_losses_arr, axis=0)
    min_losses = np.min(all_losses_arr, axis=0)
    max_losses = np.max(all_losses_arr, axis=0)

    epochs = np.arange(len(mean_losses))

    plt.figure(figsize=(10, 6))

    # Plot all individual loss lines (optional)
    for run_losses in all_losses_arr:
        plt.plot(epochs, run_losses, color='gray', alpha=0.3)

    # Plot mean loss
    plt.plot(epochs, mean_losses, color='blue', label=f'Mean {metric}')

    # Shade min/max region
    plt.fill_between(
        epochs,
        min_losses,
        max_losses,
        color='blue',
        alpha=0.2,
        label='Min/Max Range'
    )

    plt.xlabel('Epoch', size=24)
    plt.ylabel(f'{metric}', size=24)
    #plt.title(f'{metric} (Final Mean {mean_losses[-1]:.2f}) Over Epochs')
    plt.legend(fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(True)
    plt.ylim(0, all_losses_arr.max())
    if save:
        plt.savefig(save)

    plt.show()


