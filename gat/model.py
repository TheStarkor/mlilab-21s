from layers import GraphConvolutionLayer
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolutionLayer, GraphAttentionLayer


class GCN(nn.Module):
    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float) -> None:
        super(GCN, self).__init__()

        self.gc1 = GraphConvolutionLayer(nfeat, nhid)
        self.gc2 = GraphConvolutionLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        alpha: float,
        nheads: int,
    ) -> None:
        """Dense version of GAT."""
        pass
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions: List = [
            # GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            # for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module(f"attention_{i}", attention)

        # self.out_att = GraphAttentionLayer(
        #     nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False
        # )

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        pass
