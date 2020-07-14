import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import HybridizationType, MolFromSmiles
from dgl import *
from evaluate import *
from dgl.nn.pytorch import DenseGraphConv, DenseSAGEConv, DenseChebConv

FEATURE_DIM = 63
OFFSET = torch.tensor([0, 25, 7, 4, 7, 5, 6, 6, 1,]).cumsum(dim=0)
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
class GCNPoint():
    def __init__(self, n, adj, vec):
        self.n, self.adj, self.vec = n, adj, vec

def atom_feature(atom):
    index = torch.tensor([
        ATOM[atom.GetAtomicNum()], #25
        atom.GetExplicitValence(),  #7
        atom.GetImplicitValence(),  #4
        HYBRID[atom.GetHybridization()],  #7
        atom.GetTotalNumHs(),  #5
        atom.GetDegree(),  #6
        atom.GetFormalCharge()+2,  #6
        int(atom.IsInRing()),  #1
        int(atom.GetIsAromatic())]) + OFFSET  #1+1
    vec = torch.zeros(FEATURE_DIM)
    vec[index] = 1.0
    vec[-1] = atom.GetMass()/100
    return vec, FEATURE_DIM

class Models(torch.nn.Module):
    def __init__(self, device = "cpu" ,dim = 64, model_type = "dense_gcn", cheb_k = 3): #初始化
        super().__init__()
        self.dim = dim
        self.device = device
        if model_type == "dense_gcn":
            self.embed = DenseGraphConv(FEATURE_DIM, dim)
            self.conv = DenseGraphConv(dim, dim)
        elif model_type == "dense_sage":
            self.embed = DenseSAGEConv(FEATURE_DIM, dim)
            self.conv = DenseSAGEConv(dim, dim)
        elif model_type == "dense_cheb":
            print("dense_cheb")
            self.embed = DenseChebConv(FEATURE_DIM, dim, cheb_k)
            self.conv = DenseChebConv(dim, dim, cheb_k)
        self.func = torch.nn.Linear(dim, 2)
        self.act = torch.nn.ReLU()

    def process(self, smiles): #构图
        mol = MolFromSmiles(smiles)
        n = mol.GetNumAtoms()+1
        graph = DGLGraph()
        graph.add_nodes(n)
        graph.add_edges(graph.nodes(), graph.nodes())
        graph.add_edges(range(1, n), 0)
        for e in mol.GetBonds():
            u, v = e.GetBeginAtomIdx(), e.GetEndAtomIdx()
            graph.add_edge(u+1, v+1)
            graph.add_edge(v+1, u+1)
        adj = graph.adjacency_matrix(transpose=False).to_dense()
        v, m = torch.cat([atom_feature(atom)[0][None, :] for atom in mol.GetAtoms()]), FEATURE_DIM
        vec = torch.cat([torch.zeros((1, m)),v]).to(self.device)
        return GCNPoint(n, adj, vec)

    def forward(self, batch):
        result = torch.zeros((len(batch), 2))
        for i, data in enumerate(batch):
            x = self.act(self.embed(data.adj, data.vec))
            y = self.act(self.conv(data.adj, x))            #使用预先构建的 DenseGraphConv 模块获得标准的两层GCN模型
            result[i, :] = self.func(x[0] + y[0])
        return result

    def predict(self, batch):
        with torch.no_grad():
            result = self.forward(batch).softmax(dim=1)
        return result