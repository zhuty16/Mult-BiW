import numpy as np
import scipy.sparse as sp


def get_item_weight(train_dict, num_item, beta, max_capping, alpha):
    item_pop = np.zeros(num_item, dtype=np.float32)
    for u in train_dict:
        for i in train_dict[u]:
            item_pop[i] += 1.0
    item_pop_mean = np.mean(item_pop ** beta)
    item_pop_inverse = 1 / (alpha * item_pop_mean + (1 - alpha) * item_pop ** beta)
    item_pop_inverse_clip = np.array([min(p, max_capping) for p in item_pop_inverse])
    return item_pop_inverse_clip


def get_rating_matrix_sparse(train_dict, validate_dict, num_user, num_item):
    row_train = [u for u in train_dict for i in train_dict[u]]
    col_train = [i for u in train_dict for i in train_dict[u]]
    row_validate = [u for u in validate_dict for i in validate_dict[u]]
    col_validate = [i for u in validate_dict for i in validate_dict[u]]
    rating_matrix_sparse_validate = sp.csr_matrix(([1] * len(row_train), (row_train, col_train)), (num_user, num_item)).astype(np.float32)
    rating_matrix_sparse_test = sp.csr_matrix(([1] * len(row_train + row_validate), (row_train + row_validate, col_train + col_validate)), (num_user, num_item)).astype(np.float32)
    return rating_matrix_sparse_validate, rating_matrix_sparse_test


def get_user_batch(num_user, batch_size):
    user_batch = list()
    user_list = list(range(num_user))
    np.random.shuffle(user_list)
    i = 0
    while i < len(user_list):
        user_batch.append(np.array(user_list[i:i + batch_size]))
        i += batch_size
    return user_batch


def get_top_K_index(pred_scores, K):
    ind = np.argpartition(pred_scores, -K)[:, -K:]
    arr_ind = pred_scores[np.arange(len(pred_scores))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_scores)), ::-1]
    batch_pred_list = ind[np.arange(len(pred_scores))[:, None], arr_ind_argsort]
    return batch_pred_list.tolist()
