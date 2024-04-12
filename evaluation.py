import lib
import numpy as np
import torch
import pickle
from tqdm import tqdm

class Evaluation(object):
    def __init__(self, model, loss_func, use_cuda, k=20):
        self.model = model
        self.loss_func = loss_func
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def eval(self, eval_data, batch_size):
        self.model.eval()
        losses = []
        recalls = []
        mrrs = []
        accuracys = []
        ndcgs = []
        dataloader = lib.DataLoader(eval_data, batch_size)
        with torch.no_grad():
            hidden = self.model.init_hidden()
            for ii, (input, target, mask) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters = 1000):
                input = input.to(self.device)
                target = target.to(self.device)
                evidences, evidence_a, loss = self.model(input, hidden, target, 1)
                recall, mrr, accuracy, ndcg = lib.evaluate(evidence_a, target, k=self.topk)

                losses.append(loss.item())
                recalls.append(recall)
                mrrs.append(mrr)
                accuracys.append(accuracy)
                ndcgs.append(ndcg)

        mean_losses = np.mean(losses)
        mean_recall = np.mean(recalls)
        mean_mrr = np.mean(mrrs)
        mean_accuracy = np.mean(accuracys)
        mean_ndcg = np.mean(ndcgs)


        return mean_losses, mean_recall, mean_mrr, mean_accuracy, mean_ndcg


