import numpy as np
import csv


def evaluate(rank_list, test_dict, K):
    rank_list = np.array(rank_list)[:, :K]  # [batch_size, topK]
    precision_list = list()
    recall_list = list()
    ndcg_list = list()
    for user in range(rank_list.shape[0]):
        if user in test_dict and len(test_dict[user]) > 0:
            hit = len(set(rank_list[user].tolist()) & set(test_dict[user]))
            precision_list.append(hit / K)
            recall_list.append(hit / len(test_dict[user]))
            index = np.arange(len(rank_list[user].tolist()))
            k = min(len(rank_list[user].tolist()), len(test_dict[user]))
            idcg = (1 / np.log2(2 + np.arange(k))).sum()
            dcg = (1 / np.log2(2 + index[np.isin(rank_list[user].tolist(), test_dict[user])])).sum()
            ndcg_list.append(dcg/idcg)
    precision_avg = np.mean(precision_list)
    recall_avg = np.mean(recall_list)
    ndcg_avg = np.mean(ndcg_list)
    print('precision@{K}: %.4f, recall@{K}: %.4f, ndcg@{K}: %.4f'.format(K=K) % (precision_avg, recall_avg, ndcg_avg))
    return [precision_avg, recall_avg, ndcg_avg]


def save_result(args, result_valid, result_test):
    ndcg_20 = list(np.array(result_valid)[:, 6])
    ndcg_20_max = max(ndcg_20)
    result_report = result_test[ndcg_20.index(ndcg_20_max)]
    #  We get the epoch of the best results on the validation set, and report the results of that epoch on the test set.
    args_dict = vars(args)
    filename = ""
    for arg in args_dict:
        filename += str(args_dict[arg]) + "_"
    with open(filename + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "Precision@10", "Recall@10", "NDCG@10", "Precision@20", "Recall@20", "NDCG@20"])
        for line in result_test:
            writer.writerow(line)
        writer.writerow(result_report)
        for arg in args_dict:
            writer.writerow(["", arg, args_dict[arg]] + [""] * (len(line) - 3))
