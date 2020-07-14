import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import HybridizationType, MolFromSmiles
from dgl import *
from evaluate import *
from dgl.nn.pytorch import GraphConv, GATConv, GINConv, SAGEConv

ATOM = {
    6: 0,   8: 1,   7: 2,   15: 3,  16: 4,  17: 5,  11: 6,  9: 7,   35: 8,  51: 9,
    19: 10, 20: 11, 64: 12, 53: 13, 3: 14,  83: 15, 33: 16, 80: 17, 30: 18, 14: 19,
    82: 20, 26: 21, 78: 22, 34: 23, 27: 24
}
HYBRID = {
    HybridizationType.S: 0,
    HybridizationType.SP: 1,
    HybridizationType.SP2: 2,
    HybridizationType.SP3: 3,
    HybridizationType.SP3D: 4,
    HybridizationType.SP3D2: 5,
    HybridizationType.UNSPECIFIED: 6
}
class GNNPoint():
    def __init__(self, n, adj, vec):
        self.n, self.adj, self.vec = n, adj, vec

class Model(nn.Module):
    def __init__(self, device = 'cpu' ,dim = 64, model_type = "gcn"): #初始化
        super().__init__()
        self.dim = dim
        self.device = device
        self.embed = nn.Embedding(500, dim)
        if model_type == "gcn":
            self.conv = GraphConv(dim, dim)
        elif model_type == "gat":
            self.conv = GATConv(dim, dim , 1)
        elif model_type == "gin":
            print("gin")
            self.conv = GINConv(apply_func = nn.Linear(dim,dim), aggregator_type = "mean")
        elif model_type == "sage":
            print("sage")
            self.conv = SAGEConv(dim, dim, aggregator_type = "gcn")
        self.func = nn.Linear(dim, 2)
        self.act = nn.ReLU()

    def process(self, smiles): #构图
        mol = MolFromSmiles(smiles)
        n = mol.GetNumAtoms()
        graph = DGLGraph()
        graph.add_nodes(n)
        graph.add_edges(graph.nodes(), graph.nodes())
        graph.add_edges(range(1, n), 0)
        graph.ndata["element"] = torch.tensor([ATOM[atom.GetAtomicNum()] for atom in mol.GetAtoms()])
        graph.ndata["explicit"] = torch.tensor([atom.GetExplicitValence() for atom in mol.GetAtoms()])
        graph.ndata["implicit"] = torch.tensor([atom.GetImplicitValence() for atom in mol.GetAtoms()])
        graph.ndata["hybrid"] = torch.tensor([HYBRID[atom.GetHybridization()] for atom in mol.GetAtoms()])
        graph.ndata["hcount"] = torch.tensor([atom.GetTotalNumHs() for atom in mol.GetAtoms()])
        graph.ndata["degree"] = torch.tensor([atom.GetDegree() for atom in mol.GetAtoms()])
        graph.ndata["charge"] = torch.tensor([atom.GetFormalCharge() + 2 for atom in mol.GetAtoms()])
        graph.ndata["ring"] = torch.tensor([int(atom.IsInRing()) for atom in mol.GetAtoms()])
        graph.ndata["aromatic"] = torch.tensor([int(atom.GetIsAromatic()) for atom in mol.GetAtoms()])
        for e in mol.GetBonds():
            u, v = e.GetBeginAtomIdx(), e.GetEndAtomIdx()
            graph.add_edge(u, v)
            graph.add_edge(v, u)

        vec = self.embed(graph.ndata["element"] + graph.ndata["explicit"] + graph.ndata["implicit"] + graph.ndata["hybrid"] +
                       graph.ndata["hcount"] + graph.ndata["degree"] + graph.ndata["charge"] + graph.ndata["ring"] + graph.ndata["aromatic"])
        return GNNPoint(n, graph, vec)

    def forward(self, batch):
        result = torch.zeros((len(batch), 2))
        for i, data in enumerate(batch):
            x = self.act(self.conv(data.adj, data.vec))
            y = self.act(self.conv(data.adj, x))
            result[i, :] = self.func(x[0] + y[0])
        return result

    def predict(self, batch):
        with torch.no_grad():
            result = self.forward(batch).softmax(dim=1)
        return result