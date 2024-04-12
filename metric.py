import torch


def get_recall(indices, targets): #recall --> wether next item in session is within top K=20 recommended items or not

    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = float(n_hits) / targets.size(0)
    return recall


def get_mrr(indices, targets): #Mean Receiprocal Rank --> Average of rank of next item in the session.

    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).item() / targets.size(0)
    return mrr


def get_accuracy(indices, targets):

    predictions = indices[:, 0]  # Take the first predicted item as the recommendation
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy


def get_ndcg(indices, targets):

    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()

    gains = torch.reciprocal(torch.log2(ranks + 1))

    ideal_ranks = torch.arange(targets.size(1), dtype=torch.float32) + 1
    ideal_gains = torch.reciprocal(torch.log2(ideal_ranks + 1))

    dcg = torch.sum(gains)
    ideal_dcg = torch.sum(ideal_gains)

    ndcg = dcg / ideal_dcg
    ndcg = torch.mean(ndcg).item()

    return ndcg


def evaluate(indices, targets, k=20):

    _, indices = torch.topk(indices, k, -1)
    recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)
    accuracy = get_accuracy(indices, targets)
    ndcg = get_ndcg(indices, targets)
    return recall, mrr, accuracy, ndcg
