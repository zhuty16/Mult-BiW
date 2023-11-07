import time
import argparse
import numpy as np
import scipy.sparse as sp
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from evaluate import evaluate, save_result
from utils import get_item_weight, get_rating_matrix_sparse, get_user_batch, get_top_K_index


class GCN(object):
    def __init__(self, num_user, num_item, args):
        self.num_user = num_user
        self.num_item = num_item

        self.num_factor = args.num_factor
        self.l2_reg = args.l2_reg
        self.lr = args.lr
        self.num_layer = args.num_layer

        with tf.name_scope('reconstruction'):
            self.adj_matrix = tf.sparse_placeholder(tf.float32, name='adj_matrix')
            self.u = tf.placeholder(tf.int32, [None], name='u')
            self.input_u = tf.placeholder(tf.float32, [None, self.num_item], name='input_u')
            self.item_weight = tf.placeholder(tf.float32, [self.num_item], name='item_weight')

            embedding = tf.Variable(tf.random_normal([self.num_user + self.num_item, self.num_factor], stddev=0.01, name='embedding'))
            embedding_final = [embedding]
            layer = embedding
            for k in range(self.num_layer):
                layer = tf.sparse_tensor_dense_matmul(self.adj_matrix, layer)
                embedding_final += [layer]
            embedding_final = tf.stack(embedding_final, 1)
            embedding_final = tf.reduce_mean(embedding_final, 1)
            user_embedding, item_embedding = tf.split(embedding_final, [self.num_user, self.num_item], 0)

            W_in = user_embedding
            W_out = tf.transpose(item_embedding)
            b_out = tf.Variable(tf.zeros([1, self.num_item]), name='b_out')

            u_emb = tf.nn.embedding_lookup(W_in, self.u)
            input_u_hat = tf.matmul(u_emb, W_out) + b_out  # [batch_size, num_item]

        with tf.name_scope('loss'):
            loss_ce = -tf.reduce_mean(tf.reduce_sum(self.input_u * tf.nn.log_softmax(input_u_hat) * tf.expand_dims(self.item_weight, 0), 1))
            self.loss = loss_ce + self.l2_reg * tf.reduce_sum([tf.nn.l2_loss(va) for va in tf.trainable_variables()])
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.name_scope('test'):
            self.test_logits = input_u_hat - 2 ** 32 * self.input_u


def get_adj_matrix(train_dict, num_user, num_item):
    row = np.array([u for u in train_dict for i in train_dict[u]] + [i + num_user for u in train_dict for i in train_dict[u]])
    col = np.array([i + num_user for u in train_dict for i in train_dict[u]] + [u for u in train_dict for i in train_dict[u]])
    rel_matrix = sp.coo_matrix(([1.0] * len(row), (row, col)), shape=(num_user + num_item, num_user + num_item)).astype(np.float32)
    rowsum = np.array(rel_matrix.sum(1)) + 1e-24
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    rel_matrix_normalized = degree_mat_inv_sqrt.dot(rel_matrix.dot(degree_mat_inv_sqrt)).tocoo()
    indices = np.vstack((rel_matrix_normalized.row, rel_matrix_normalized.col)).transpose()
    values = rel_matrix_normalized.data.astype(np.float32)
    shape = rel_matrix_normalized.shape
    return indices, values, shape


def get_adj_matrix_test(train_dict, validate_dict, num_user, num_item):
    row = np.array([u for u in train_dict for i in train_dict[u]] + [i + num_user for u in train_dict for i in train_dict[u]] + [u for u in validate_dict for i in validate_dict[u]] + [i + num_user for u in validate_dict for i in validate_dict[u]])
    col = np.array([i + num_user for u in train_dict for i in train_dict[u]] + [u for u in train_dict for i in train_dict[u]] + [i + num_user for u in validate_dict for i in validate_dict[u]] + [u for u in validate_dict for i in validate_dict[u]])
    rel_matrix = sp.coo_matrix(([1.0] * len(row), (row, col)), shape=(num_user + num_item, num_user + num_item)).astype(np.float32)
    rowsum = np.array(rel_matrix.sum(1)) + 1e-24
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    rel_matrix_normalized = degree_mat_inv_sqrt.dot(rel_matrix.dot(degree_mat_inv_sqrt)).tocoo()
    indices = np.vstack((rel_matrix_normalized.row, rel_matrix_normalized.col)).transpose()
    values = rel_matrix_normalized.data.astype(np.float32)
    shape = rel_matrix_normalized.shape
    return indices, values, shape


def get_feed_dict(model, user_batch, rating_matrix_sparse, item_weight, adj_matrix):
    feed_dict = dict()
    feed_dict[model.u] = np.array(user_batch)
    feed_dict[model.input_u] = np.array(rating_matrix_sparse[list(user_batch)].todense())
    feed_dict[model.item_weight] = item_weight
    feed_dict[model.adj_matrix] = adj_matrix
    return feed_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LightGCN')
    parser.add_argument('--dataset', type=str, default='amazon')
    # hyperparameters for GCN
    parser.add_argument('--num_factor', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2_reg', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epoch', type=int, default=400)
    parser.add_argument('--random_seed', type=int, default=2023)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument("--N", type=int, default=10)
    # hyperparameters for reweighting
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max_capping", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=-1.0)  # [0.0, 1.0] U {-1.0}
    args = parser.parse_args()
    for arg, arg_value in vars(args).items():
        print(arg, ":", arg_value)

    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    [train_dict, validate_dict, test_dict, num_user, num_item] = np.load("data/{dataset}/{dataset}.npy".format(dataset=args.dataset), allow_pickle=True)
    print("num_user: %d, num_item: %d" % (num_user, num_item))
    train_dict_len = [len(train_dict[u]) for u in train_dict]
    print("max len: %d, min len: %d, avg len: %.4f, med len: %.4f" % (np.max(train_dict_len), np.min(train_dict_len), np.mean(train_dict_len), np.median(train_dict_len)))
    rating_matrix_sparse_validate, rating_matrix_sparse_test = get_rating_matrix_sparse(train_dict, validate_dict, num_user, num_item)

    adj_matrix = get_adj_matrix(train_dict, num_user, num_item)
    adj_matrix_test = get_adj_matrix_test(train_dict, validate_dict, num_user, num_item)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print("Model preparing...")
        model = GCN(num_user, num_item, args)
        sess.run(tf.global_variables_initializer())

        print("Model training...")
        result_validate = list()
        result_test = list()
        for epoch in range(1, args.num_epoch + 1):
            t1 = time.time()
            alpha = 1 - (epoch / args.num_epoch) ** args.eta if args.alpha == -1.0 else args.alpha
            item_weight = get_item_weight(train_dict, num_item, args.beta, args.max_capping, alpha)

            train_loss = list()
            user_batch = get_user_batch(num_user, args.batch_size)
            for batch in user_batch:
                loss, _ = sess.run([model.loss, model.train_op], feed_dict=get_feed_dict(model, batch, rating_matrix_sparse_validate, item_weight, adj_matrix))
                train_loss.append(loss)
            train_loss = np.mean(train_loss)
            print("epoch: %d, %.2fs" % (epoch, time.time() - t1))
            print("training loss: %.4f" % train_loss)

            if epoch == 1 or epoch % args.N == 0:
                batch_size_test = args.batch_size
                rank_list = list()
                for start in range(0, num_user, batch_size_test):
                    test_logits = sess.run(model.test_logits, feed_dict=get_feed_dict(model, np.arange(start, min(start + batch_size_test, num_user)), rating_matrix_sparse_validate, item_weight, adj_matrix))
                    rank_list += get_top_K_index(test_logits, 20)
                rank_list = np.array(rank_list)
                result_validate.append([epoch] + evaluate(rank_list, validate_dict, 10) + evaluate(rank_list, validate_dict, 20))

                rank_list = list()
                for start in range(0, num_user, batch_size_test):
                    test_logits = sess.run(model.test_logits, feed_dict=get_feed_dict(model, np.arange(start, min(start + batch_size_test, num_user)), rating_matrix_sparse_test, item_weight, adj_matrix_test))
                    rank_list += get_top_K_index(test_logits, 20)
                rank_list = np.array(rank_list)
                result_test.append([epoch] + evaluate(rank_list, test_dict, 10) + evaluate(rank_list, test_dict, 20))
                #  We get the epoch of the best results on the validation set, and report the results of that epoch on the test set.

        save_result(args, result_validate, result_test)
