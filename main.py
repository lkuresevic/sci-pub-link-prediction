import torch
from torch_geometric.data import Data

from data_prep import *
from network_classes.gcn_encoder import GCN_Encoder
from utils import *
from train import *
from eval import *
from constants import *

if __name__ == '__main__':
    set_seed()
    
    # Load Cora    
    dataset = load_cora()
    graph = dataset[0]
    
    # Split dataset
    train_data, val_data, test_data = split_data(graph)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)

    model = GCN_Encoder(dataset.num_features, 128, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    model = train_link_predictor(model, train_data, val_data, optimizer, criterion)

    test_auc = eval_link_predictor(model, test_data)
    print(f"Test: {test_auc:.3f}")


