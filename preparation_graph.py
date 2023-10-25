"""
Edge to Node Project.
    Preparation Graph
"""
import pickle

import networkx as nx
import pandas
import torch
from torch_geometric.data import Data

if __name__ == '__main__':
    TRAIN_POS_EDGES = pandas.read_csv('/mnt/disk2/ali.rahmati/link2node/data/Raw/ddi/pos_train.csv')
    TRAIN_NEG_EDGES = pandas.read_csv('/mnt/disk2/ali.rahmati/link2node/data/Raw/ddi/neg_train.csv')

    TRAIN_POS_EDGES = [(a, b) for a, b in zip(TRAIN_POS_EDGES['node_1'], TRAIN_POS_EDGES['node_2'])]
    TRAIN_NEG_EDGES = [(a, b) for a, b in zip(TRAIN_NEG_EDGES['node_1'], TRAIN_NEG_EDGES['node_2'])]

    TEST_POS_EDGES = pandas.read_csv('/mnt/disk2/ali.rahmati/link2node/data/Raw/ddi/pos_test.csv')
    TEST_NEG_EDGES = pandas.read_csv('/mnt/disk2/ali.rahmati/link2node/data/Raw/ddi/neg_test.csv')

    TEST_POS_EDGES = [(a, b) for a, b in zip(TEST_POS_EDGES['node_1'], TEST_POS_EDGES['node_2'])]
    TEST_NEG_EDGES = [(a, b) for a, b in zip(TEST_NEG_EDGES['node_1'], TEST_NEG_EDGES['node_2'])]

    VAL_POS_EDGES = pandas.read_csv('/mnt/disk2/ali.rahmati/link2node/data/Raw/ddi/pos_val.csv')
    VAl_NEG_EDGES = pandas.read_csv('/mnt/disk2/ali.rahmati/link2node/data/Raw/ddi/neg_val.csv')

    VAL_POS_EDGES = [(a, b) for a, b in zip(VAL_POS_EDGES['node_1'], VAL_POS_EDGES['node_2'])]
    VAL_NEG_EDGES = [(a, b) for a, b in zip(VAl_NEG_EDGES['node_1'], VAl_NEG_EDGES['node_2'])]

    G = nx.Graph()
    G.add_edges_from(TRAIN_POS_EDGES, negative=False, label='train')
    G.add_edges_from(TRAIN_NEG_EDGES, negative=True, label='train')

    G.add_edges_from(TEST_POS_EDGES, negative=False, label='test')
    G.add_edges_from(TEST_NEG_EDGES, negative=True, label='test')

    G.add_edges_from(VAL_POS_EDGES, negative=False, label='val')
    G.add_edges_from(VAL_NEG_EDGES, negative=True, label='val')

    H = nx.Graph()

    for node in G.nodes():
        H.add_node(node, label="0", mask=None)

    k = len(H.nodes)

    for edge in G.edges():
        u, v = edge[0], edge[1]
        H.add_edge(u, v)
        if G.edges[u, v]['negative'] is True:
            H.add_node(k, label="1", mask=G.edges[u, v]['label'])

        else:
            H.add_node(k, label="2", mask=G.edges[u, v]['label'])

        H.add_edge(u, k)
        H.add_edge(v, k)
        k += 1

    print('new graph details : ', H)

    LABEL_TO_NUMERIC = {"0": 0, "1": 1, "2": 2}
    NODE_LABELS = [LABEL_TO_NUMERIC[H.nodes[node]['label']] for node in H.nodes]

    X = torch.tensor(NODE_LABELS, dtype=torch.float)
    EDGE_INDEX = torch.tensor(list(H.edges), dtype=torch.long).t().contiguous()

    TRAIN_MASK = torch.tensor([H.nodes[node]['mask'] == 'train' for node in H.nodes()], dtype=torch.bool)
    TEST_MASK = torch.tensor([H.nodes[node]['mask'] == 'test' for node in H.nodes()], dtype=torch.bool)
    VAL_MASK = torch.tensor([H.nodes[node]['mask'] == 'val' for node in H.nodes()], dtype=torch.bool)

    PYG_DATA = Data(num_nodes=len(H.nodes()), edge_index=EDGE_INDEX,
                    train_mask=TRAIN_MASK, test_mask=TEST_MASK, val_mask=VAL_MASK,
                    label=NODE_LABELS)
    print(PYG_DATA)

    pickle.dump(PYG_DATA,
                open("/mnt/disk2/ali.rahmati/link2node/data/Processed/ddi/new_graph_pyg_with_all_mask_ddi.pickle",
                     "wb"))
