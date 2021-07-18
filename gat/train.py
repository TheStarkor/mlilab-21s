import time
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from option import args
from utils import load_data, accuracy
from model import GCN, GAT


if __name__ == "__main__":

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print(args)
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    if args.model == "GCN":
        model = GCN(
            nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
        )
    else:
        model = GAT(
            nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            nheads=args.nb_nheads,
            alpha=args.alpha
        )
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    t_total = time.time()

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            model.eval()
            output = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        print(
            f"Epoch: {epoch+1:04d}",
            f"loss_train: {loss_train.item():.4f}",
            f"acc_train: {acc_train.item():.4f}",
            f"loss_val: {loss_val.item():.4f}",
            f"acc_val: {acc_val.item():.4f}",
            f"time: {time.time() - t:.4f}s",
        )

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
