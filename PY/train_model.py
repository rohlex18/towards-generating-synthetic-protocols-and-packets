import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PY.TrafficClassifier import TrafficClassifier
from PY.PacketDataset import PacketDataset
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from PY.Plotting import plot_protocol_distribution
from sklearn.metrics import recall_score
import torch.nn.functional as F


def train_model(X, y, ps, batch_size = 2, num_pkts = 256, pkt_len = 32, plots=False, val=False):
    train = list(set(ps))
    if not val:
        val = random.sample(train, 2)
    # Remove them from the original list
    for item in val:
        train.remove(item)


    ps = np.array(ps)
    train_inds =  np.where(np.isin(ps, train))[0]
    val_inds =  np.where(np.isin(ps, val))[0]

    trainX = [X[i] for i in train_inds]
    trainY = [y[i] for i in train_inds]
    trainP = [ps[i] for i in train_inds]

    valX = [X[i] for i in val_inds]
    valY = [y[i] for i in val_inds]
    valP = [ps[i] for i in val_inds]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")


    dataset = PacketDataset(trainX, trainY, protocols=trainP, pkt_len=pkt_len, num_pkts = num_pkts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_data = PacketDataset(valX, valY, protocols=valP, pkt_len=pkt_len, num_pkts = num_pkts)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    if plots:
        plot_protocol_distribution(loader)
        plot_protocol_distribution(val_loader)


    losses = []
    val_scores = []
    model = TrafficClassifier(pkt_len).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50

    for epoch in range(num_epochs):
        #print(f"Epoch {epoch}")
        i = 0
        model.train()  # set to training mode

        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if i % 10 == 0:  # optional: print every 10 batches
                #print(f"Batch {i}: loss = {loss.item():.4f}")
            i += 1

        losses.append(loss.item()) #last loss only?

        # Validation after each epoch
        model.eval()  # set to evaluation mode

        from sklearn.metrics import f1_score

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model(inputs)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / len(all_labels) if all_labels else 0.0

        val_scores.append(accuracy)



        #print(f"Epoch {epoch}, Training Loss: {losses[-1]:.2f} Validation loss: {avg_val_loss:.2f}, accuracy: {accuracy:.2f}")

    # Plot training loss and validation accuracy
    if plots:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(val_scores, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

    return model, losses, val_scores
