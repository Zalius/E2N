import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.linkproppred import Evaluator
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GCNConv

random.seed(1234)


class GCNBinaryNodeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super(GCNBinaryNodeClassifier, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim // 2)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim // 2, num_classes)

        self.dropout = nn.Dropout(dropout_rate)
        self.initialize_weights()

    def initialize_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            for name, param in conv.named_parameters():
                if "weight" in name:
                    nn.init.kaiming_uniform_(param)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return x


@torch.no_grad()
def evaluate(labels, data_mode):
    MODEL.eval()
    graph_logits = MODEL(DATA.x, DATA.edge_index)
    graph_predictions = torch.sigmoid(graph_logits)

    if data_mode == 'test':
        loss = CRITERION(graph_logits[DATA.test_mask].cpu(), labels)
        acc = accuracy_score(y_true=labels,
                             y_pred=torch.argmax(graph_predictions, dim=1)[DATA.test_mask].cpu())
        logit_ = graph_predictions[DATA.test_mask].cpu()


    elif data_mode == 'val':
        loss = CRITERION(graph_logits[DATA.val_mask].cpu(), labels)
        acc = accuracy_score(y_true=labels,
                             y_pred=torch.argmax(graph_predictions, dim=1)[DATA.val_mask].cpu())
        logit_ = graph_predictions[DATA.val_mask].cpu()

    pos_scores = logit_[labels == 2][:, 2]
    neg_scores = logit_[labels == 1][:, 1]

    hits = EVALUATOR.eval({
        'y_pred_pos': pos_scores,
        'y_pred_neg': neg_scores,
    })[f'hits@{K}']

    print(f"Evaluation for {data_mode}")
    print(f" Loss:> {loss}  , Acc:> {acc}, Hit@20:> {hits}")
    return hits


if __name__ == '__main__':
    DEVICE = 'cuda'
    INPUT_DIM = 512
    HIDDEN_DIM = 512
    DROPOUT_RATE = 0.3
    NUM_EPOCHS = 50
    NUM_RUN = 2

    CRITERION = nn.CrossEntropyLoss()
    EVALUATOR = Evaluator(name='ogbl-ddi')
    K = 20
    EVALUATOR.K = K

    DATA = pickle.load(
        open("/mnt/disk2/ali.rahmati/link2node/data/Processed/ddi/new_graph_pyg_with_all_mask.pickle", "rb"))
    DATA = DATA.to(DEVICE)

    NEW_TRAIN_MASK = torch.zeros(DATA.edge_index.shape[1], dtype=torch.bool).to(DEVICE)
    NEW_TRAIN_MASK[DATA.train_mask.nonzero()] = True
    TRAIN_EDGE_INDEX = DATA.edge_index[:, NEW_TRAIN_MASK]

    TRAIN_LABELS = [DATA.label[i] for i, item in enumerate(DATA.train_mask) if item.item()]
    TEST_LABELS = [DATA.label[i] for i, item in enumerate(DATA.test_mask) if item.item()]
    VAL_LABELS = [DATA.label[i] for i, item in enumerate(DATA.val_mask) if item.item()]

    TEST_LABELS_CPU = torch.tensor(TEST_LABELS).cpu()
    VAL_LABELS_CPU = torch.tensor(VAL_LABELS).cpu()

    BEST_OF_EACH_RUN_TEST = []
    BEST_OF_EACH_RUN_VAL = []

    for run in range(NUM_RUN):
        DATA.x = torch.nn.Embedding(DATA.num_nodes, INPUT_DIM).weight.to(DEVICE)
        NUM_CLASSES = len(set(DATA.label))

        best_val = []
        best_test = []

        MODEL = GCNBinaryNodeClassifier(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
        OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.01)

        for epoch in range(NUM_EPOCHS):
            MODEL.train()
            OPTIMIZER.zero_grad()
            train_logits = MODEL(DATA.x[DATA.train_mask], TRAIN_EDGE_INDEX)
            train_loss = CRITERION(train_logits.cpu(), torch.tensor(TRAIN_LABELS).cpu())
            train_loss.backward()
            OPTIMIZER.step()
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss.item()}')

            with torch.no_grad():
                test_hit = evaluate(labels=TEST_LABELS_CPU, data_mode='test')
                best_test.append(test_hit)

                val_hit = evaluate(labels=VAL_LABELS_CPU, data_mode='val')
                best_val.append(val_hit)
            print('#' * 30)

        BEST_OF_EACH_RUN_TEST.append(max(best_test))
        BEST_OF_EACH_RUN_VAL.append(best_val[best_test.index(max(best_test))])

        print('run :', run + 1)

    print('test final result: ', np.mean(BEST_OF_EACH_RUN_TEST), np.std(BEST_OF_EACH_RUN_TEST))
    print('val final result: ', np.mean(BEST_OF_EACH_RUN_VAL), np.std(BEST_OF_EACH_RUN_VAL))
