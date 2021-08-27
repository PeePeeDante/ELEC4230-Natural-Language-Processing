import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from RNN import RNN_M2O
from copy import deepcopy
from dataloader import get_dataloaders
from RNN import RNN_M2O


def trainer(train_loader, dev_loader, model, optimizer, criterion, early_stop, epoch=1000):
    
    early_stop_counter = early_stop
    best_perplexity = 100000000
    best_model = deepcopy(model)

    for e in range(epoch):
        
        print("(Epoch {}) TRAIN".format((e + 1)))
        loss_log = []
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for _ , (X, Y) in pbar:

            optimizer.zero_grad()
            softmax_probs = model(X)
            loss = criterion(softmax_probs,Y)
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())

        # model evaluation
        print("(Epoch {}) EVALUTATE".format((e + 1)))
        model.eval()
        eval_loss = []
        for _, (X, Y) in tqdm(enumerate(dev_loader),total = len(dev_loader)):
            logit = model(X)
            loss = criterion(logit,Y)
            eval_loss.append(loss.item())
        
        perplexity = np.exp(np.mean(eval_loss))

        if best_perplexity > perplexity:
            best_perplexity = perplexity
            best_model = deepcopy(model)
            early_stop_counter = early_stop
        else:
            early_stop_counter -= 1
        print("(Epoch {}) TRAIN LOSS:{:.4f}".format((e + 1), np.sum(loss_log)))
        print("(Epoch {}) EVALUATE PERPLEXITY:{:.4f} BEST PERPLEXITY:{:.4f}".format((e + 1), perplexity, best_perplexity))
        print()


        if early_stop_counter == 0:
            break

    return best_model, best_perplexity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layer_num", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # load data
    train_loader, dev_loader, vocab_num = get_dataloaders(args.batch_size)
    
    # build model
    model = RNN_M2O(args,vocab_num)
    
    # criterion function
    criterion = nn.CrossEntropyLoss()
    
    # choose optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    model, best_perplexity = trainer(train_loader, dev_loader, model, optimizer, criterion, early_stop=args.early_stop)

    print('best_dev_perplexity:{}'.format(best_perplexity))

if __name__ == "__main__":
    main()
