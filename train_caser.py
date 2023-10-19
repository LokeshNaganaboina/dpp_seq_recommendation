import argparse
from time import time
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import pickle as cPickle
import numpy as np

from caser import Caser  
from evaluation import evaluate_ranking  
from interactions import Interactions
from utils import *

tf.random.set_seed(1234)


class Recommender(object):
    """
    Contains attributes and methods that needed to train a sequential
    recommendation model. Models are trained by many tuples of
    (users, sequences, targets, negatives) and negatives are from negative
    sampling: for any known tuple of (user, sequence, targets), one or more
    items are randomly sampled to act as negatives.


    Parameters
    ----------

    n_iter: int,
        Number of iterations to run.
    batch_size: int,
        Minibatch size.
    l2: float,
        L2 loss penalty, also known as the 'lambda' of l2 regularization.
    neg_samples: int,
        Number of negative samples to generate for each targets.
        If targets=3 and neg_samples=3, then it will sample 9 negatives.
    learning_rate: float,
        Initial learning rate.
    use_cuda: boolean,
        Run the model on a GPU or CPU.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self,
                 n_iter=None,
                 batch_size=None,
                 l2=None,
                 neg_samples=None,
                 learning_rate=None,
                 use_cuda=False,
                 model_args=None):

        # model related
        self._num_items = None
        self._num_users = None
        self._net = None
        self.model_args = model_args

        # learning related
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples
        self._device = "GPU:0" if use_cuda and tf.config.experimental.list_physical_devices('GPU') else "CPU:0"

        # rank evaluation related
        self.test_sequence = None
        self._candidate = dict()

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences
        print("Value of L before creating Caser object:", self.model_args.L)
        self._net = Caser(self._num_users,
                          self._num_items,
                          self.model_args)
        
        # If using L2 regularization, it should be added in the model layers, not in the optimizer in TensorFlow.
        self._optimizer = optimizers.Adam(learning_rate=self._learning_rate)
        
    def fit(self, train, test, cate, config, verbose=False):
        """
        The general training loop to fit the model

        Parameters
        ----------

        train: Interactions (assumed to be converted to TensorFlow compatible format)
            training instances, also contains test sequences
        test: Interactions (assumed to be converted to TensorFlow compatible format)
            only contains targets for test sequences
        verbose: bool, optional
            print the logs
        """
        ##################################
        # read pre-learned kernel
        ###################################
        with open(config.l_kernel_emb, 'rb') as f:
            lk_param = cPickle.load(f)

        lk_tensor = tf.convert_to_tensor(lk_param['V'], dtype=tf.float32)
        lk_emb_i = tf.linalg.l2_normalize(lk_tensor, axis=1)
        l_kernel = tf.matmul(lk_emb_i, tf.transpose(lk_emb_i))
        l_kernel = tf.sigmoid(l_kernel)  # this line is optional

        # convert to sequences, targets and users
        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = np.reshape(train.sequences.user_ids, (-1, 1))

        L, T = train.sequences.L, train.sequences.T
        n_train = sequences_np.shape[0]
        print('total training instances:', n_train)

        if not self._initialized:
            self._initialize(train)

        for epoch_num in range(self._n_iter):
            t1 = time()

            # Shuffle data
            indices = np.arange(users_np.shape[0])
            np.random.shuffle(indices)
            users_np, sequences_np, targets_np = users_np[indices], sequences_np[indices], targets_np[indices]

            negatives_np = self._generate_negative_samples(users_np, train, n=self._neg_samples)

            # Convert numpy arrays to TensorFlow tensors
            users, sequences, targets, negatives = [tf.convert_to_tensor(x, dtype=tf.int32) for x in [users_np, sequences_np, targets_np, negatives_np]]

            print("Shape of tensor_name:", tf.shape(users))
            print("Shape of tensor_name:", tf.shape(sequences))
            print("Shape of tensor_name:", tf.shape(targets))
            print("Shape of tensor_name:", tf.shape(negatives))

            epoch_loss = 0.0

            for (minibatch_num, (batch_users, batch_sequences, batch_targets, batch_negatives)) in enumerate(minibatch(users, sequences, targets, negatives, batch_size=self._batch_size)):
                with tf.GradientTape() as tape:
                    items_to_predict = tf.concat([batch_targets, batch_negatives, batch_sequences], 1)
                    items_prediction = self._net(seq_var=batch_sequences, user_var=batch_users, item_var=items_to_predict)
                    targets_prediction, negatives_prediction, seq_prediction = tf.split(items_prediction, [tf.shape(batch_targets)[1], tf.shape(batch_negatives)[1], tf.shape(batch_sequences)[1]], axis=1)

                    if config.dpp_loss == 0:
                        # compute the binary cross-entropy loss
                        positive_loss = -tf.reduce_mean(tf.math.log(tf.sigmoid(targets_prediction)))
                        negative_loss = -tf.reduce_mean(tf.math.log(1 - tf.sigmoid(negatives_prediction)))
                        loss = positive_loss + negative_loss
                    
                    # DSL
                    elif config.dpp_loss == 1:
                        dpp_lhs = []
                        size = targets_prediction.shape[0]
                        batch_sets = tf.concat([batch_targets, batch_negatives], axis=1)
                        batch_predictions = tf.concat([targets_prediction, negatives_prediction], axis=1)
        
                        # minibatch format
                        if config.batch_format == 1:
                            batch_pos_kernel = tf.zeros([size, config.T, config.T])
                            batch_set_kernel = tf.zeros([size, config.T + config.neg_samples, config.T + config.neg_samples])
            
                            for n in range(size):
                                batch_target_index = batch_targets[n].numpy() - 1
                                batch_set_index = batch_sets[n].numpy() - 1
                                
                                batch_pos_kernel = tf.tensor_scatter_nd_update(batch_pos_kernel, [[n]], [tf.gather(tf.gather(l_kernel, batch_target_index, axis=0), batch_target_index, axis=1)])
                                batch_set_kernel = tf.tensor_scatter_nd_update(batch_set_kernel, [[n]], [tf.gather(tf.gather(l_kernel, batch_set_index, axis=0), batch_set_index, axis=1)])
                                        
                            batch_pos_q = tf.linalg.diag(tf.exp(targets_prediction))
                            batch_set_q = tf.linalg.diag(tf.exp(batch_predictions))
                            batch_pos_kernel = tf.matmul(tf.matmul(batch_pos_q, batch_pos_kernel), batch_pos_q)
                            batch_set_kernel = tf.matmul(tf.matmul(batch_set_q, batch_set_kernel), batch_set_q)
                
                            p_diag = tf.eye(config.T) * 1e-5
                            pa_diag = tf.reshape(p_diag, [1, config.T, config.T])
                            pbatch_diag = tf.repeat(pa_diag, repeats=size, axis=0)
                
                            s_diag = tf.eye(config.T + config.neg_samples)
                            sa_diag = tf.reshape(s_diag, [1, config.T + config.neg_samples, config.T + config.neg_samples])
                            sbatch_diag = tf.repeat(sa_diag, repeats=size, axis=0)
                
                            batch_pos_det = tf.linalg.det(batch_pos_kernel + pbatch_diag)
                            batch_set_det = tf.linalg.det(batch_set_kernel + sbatch_diag)
                
                            dpp_loss = tf.math.log(batch_pos_det / batch_set_det)
                            loss = -tf.reduce_mean(dpp_loss)
                        else:
                            for n in range(size):
                                pos_q = tf.linalg.diag(tf.exp(targets_prediction[n]))
                                set_q = tf.linalg.diag(tf.exp(batch_predictions[n]))
                    
                                #pos_l_kernel = l_kernel[batch_targets[n]-1][:, batch_targets[n]-1]
                                #set_l_kernel = l_kernel[batch_sets[n]-1][:, batch_sets[n]-1]

                                pos_l_kernel_indices = batch_targets[n] - 1
                                set_l_kernel_indices = batch_sets[n] - 1

                                pos_l_kernel = tf.gather(l_kernel, pos_l_kernel_indices, axis=0)
                                pos_l_kernel = tf.gather(pos_l_kernel, pos_l_kernel_indices, axis=1)

                                set_l_kernel = tf.gather(l_kernel, set_l_kernel_indices, axis=0)
                                set_l_kernel = tf.gather(set_l_kernel, set_l_kernel_indices, axis=1)
                                
                                pos_k = tf.matmul(tf.matmul(pos_q, pos_l_kernel), pos_q)
                                set_k = tf.matmul(tf.matmul(set_q, set_l_kernel), set_q)
                                
                                pos_det = tf.linalg.det(pos_k + tf.eye(tf.shape(batch_targets[n])[0]) * 1e-5)
                                set_det = tf.linalg.det(set_k + tf.eye(tf.shape(batch_sets[n])[0]))
                                
                                dpp_loss = tf.math.log(pos_det / set_det)
                                dpp_lhs.append(dpp_loss)
                            loss = -tf.reduce_mean(dpp_lhs)
                
                    #CDSL
                    elif config.dpp_loss == 2:
                        dpp_lhs = []
                        size = targets_prediction.shape[0]
                        set_items = tf.concat([batch_sequences, batch_targets, batch_negatives], axis=1)
                        set_predictions = tf.concat([seq_prediction, targets_prediction, negatives_prediction], axis=1)
                        
                        pos_items = tf.concat([batch_sequences, batch_targets], axis=1)
                        pos_predictions = tf.concat([seq_prediction, targets_prediction], axis=1)  # L+T
                        if config.batch_format == 1:
                            batch_pos_kernel = tf.zeros([size, config.L + config.T, config.L + config.T])
                            batch_set_kernel = tf.zeros([size, config.L + config.T + config.neg_samples, config.L + config.T + config.neg_samples])
                            
                            for n in range(size):
                                batch_pos_kernel = tf.tensor_scatter_nd_update(batch_pos_kernel, [[n]], [l_kernel[pos_items[n]-1][:, pos_items[n]-1]])
                                batch_set_kernel = tf.tensor_scatter_nd_update(batch_set_kernel, [[n]], [l_kernel[set_items[n]-1][:, set_items[n]-1]])
                            
                            batch_pos_q = tf.linalg.diag(tf.exp(pos_predictions))
                            batch_set_q = tf.linalg.diag(tf.exp(set_predictions))
                            
                            batch_pos_kernel = tf.matmul(tf.matmul(batch_pos_q, batch_pos_kernel), batch_pos_q)
                            batch_set_kernel = tf.matmul(tf.matmul(batch_set_q, batch_set_kernel), batch_set_q)
                            
                            p_diag = tf.eye(config.L + config.T) * 1e-3
                            pa_diag = tf.reshape(p_diag, [1, config.L + config.T, config.L + config.T])
                            pbatch_diag = tf.repeat(pa_diag, repeats=size, axis=0)
                            
                            s_diag = tf.linalg.diag(tf.concat([tf.constant([1e-3]*config.L), tf.constant([1]*(config.neg_samples+config.T))], axis=0))
                            sa_diag = tf.reshape(s_diag, [1, config.L + config.T + config.neg_samples, config.L + config.T + config.neg_samples])
                            sbatch_diag = tf.repeat(sa_diag, repeats=size, axis=0)
                            
                            batch_pos_det = tf.linalg.det(batch_pos_kernel + pbatch_diag)
                            batch_set_det = tf.linalg.det(batch_set_kernel + sbatch_diag)
                            
                            dpp_loss = tf.math.log(batch_pos_det / batch_set_det)
                            loss = -tf.reduce_mean(dpp_loss)
                        else:
                            diag_I = tf.linalg.diag(tf.concat([tf.constant([1e-3]*config.L), tf.constant([1]*(config.neg_samples+config.T))], axis=0))
                            diag_posI = tf.linalg.diag(tf.constant([1e-3]*(config.L+config.T)))
                            for n in range(size):
                                pos_q = tf.linalg.diag(tf.exp(pos_predictions[n]))
                                set_q = tf.linalg.diag(tf.exp(set_predictions[n]))
                                
                                pos_item_index = pos_items[n].numpy() - 1
                                set_item_index = set_items[n].numpy() - 1
                                    
                                pos_l_kernel = tf.gather(tf.gather(l_kernel, pos_item_index, axis=0), pos_item_index, axis=1)
                                set_l_kernel = tf.gather(tf.gather(l_kernel, set_item_index, axis=0), set_item_index, axis=1)
                                
                                pos_k = tf.matmul(tf.matmul(pos_q, pos_l_kernel), pos_q)
                                set_k = tf.matmul(tf.matmul(set_q, set_l_kernel), set_q)
                                
                                pos_det = tf.linalg.det(pos_k + diag_posI)
                                set_det = tf.linalg.det(set_k + diag_I)
                                
                                dpp_loss = tf.math.log(pos_det / set_det)
                                dpp_lhs.append(dpp_loss)
                            loss = -tf.reduce_mean(dpp_lhs)

                        epoch_loss += loss.numpy()

                # Compute gradients and apply them
                gradients = tape.gradient(loss, self._net.trainable_variables)
                self._optimizer.apply_gradients(zip(gradients, self._net.trainable_variables))
            epoch_loss /= minibatch_num + 1

            t2 = time()
            if verbose:
                if (epoch_num+1) % 10 == 0:
                    precision, recall, ndcg, cc = evaluate_ranking(self, test, config, l_kernel, cate, train, k=[3, 5, 10])
                    output_str = "Epoch %d [%.1f s], loss=%.4f, " \
                                "prec@3=%.4f, *prec@5=%.4f, prec@10=%.4f, " \
                                "recall@3=%.4f, recall@5=%.4f, recall@10=%.4f, " \
                                "ndcg@3=%.4f, ndcg@5=%.4f, ndcg@10=%.4f, " \
                                "*cc@3=%.4f, cc@5=%.4f, cc@10=%.4f, [%.1f s]" % (epoch_num + 1,
                                                                                            t2 - t1,
                                                                                            epoch_loss,
                                                                                            np.mean(precision[0]),
                                                                                            np.mean(precision[1]),
                                                                                            np.mean(precision[2]),
                                                                                            np.mean(recall[0]),
                                                                                            np.mean(recall[1]),
                                                                                            np.mean(recall[2]),
                                                                                            np.mean(ndcg[0]),
                                                                                            np.mean(ndcg[1]),
                                                                                            np.mean(ndcg[2]),
                                                                                            np.mean(cc[0]),
                                                                                            np.mean(cc[1]),
                                                                                            np.mean(cc[2]),
                                                                                            time() - t2)
                    
                    print(output_str)
            else:
                output_str = "Epoch %d [%.1f s]\tloss=%.4f [%.1f s]" % (epoch_num + 1,
                                                                        t2 - t1,
                                                                        epoch_loss,
                                                                        time() - t2)
                print(output_str)
        
    def _generate_negative_samples(self, users, interactions, n):
        
        """
        Sample negative from a candidate set of each user. The
        candidate set of each user is defined by:
        {All Items} \ {Items Rated by User}

        Parameters
        ----------

        users: array of np.int64
            sequence users
        interactions: :class:`spotlight.interactions.Interactions`
            training instances, used for generate candidates
        n: int
            total number of negatives to sample for each sequence
        """

        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not self._candidate:
            all_items = np.arange(interactions.num_items - 1) + 1  # 0 for padding
            train = interactions.tocsr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))

        for i, u in enumerate(users_):
            for j in range(n):
                x = self._candidate[u]
                negative_samples[i, j] = x[
                    np.random.randint(len(x))]

        return negative_samples

    def predict(self, user_id, item_ids=None):
        if self.test_sequence is None:
            raise ValueError('Missing test sequences, cannot make predictions')

        # set model to evaluation mode
        self._net.trainable = False

        sequences_np = self.test_sequence.sequences[user_id, :]
        sequences_np = np.atleast_2d(sequences_np)

        if item_ids is None:
            item_ids = np.arange(self._num_items).reshape(-1, 1)

        sequences = tf.convert_to_tensor(sequences_np, dtype=tf.int32)
        item_ids = tf.convert_to_tensor(item_ids, dtype=tf.int32)
        user_id = tf.convert_to_tensor(np.array([[user_id]]), dtype=tf.int32)

        print("Shape of tensor_name:", tf.shape(sequences))
        print("Shape of tensor_name:", tf.shape(item_ids))
        print("Shape of tensor_name:", tf.shape(user_id))

        # Assuming self._device is either 'CPU' or 'GPU'. Adjust accordingly.
        device = '/CPU:0' if self._device == 'CPU' else '/GPU:0'
        with tf.device(device):
            out = self._net(sequences, user_id, item_ids, for_pred=True)

        return out.numpy().flatten()
    
    def sigma(self, x):
        res = 1 - tf.exp(-model_config.sigma_alpha * x)
        return res

def get_cates_map(cate_file):
    iidcate_map = {}  # iid:cates
    ## movie_id:cate_ids, cate_ids is not only one
    with open(cate_file) as f_cate:
        for l in f_cate.readlines():
            if len(l) == 0: break
            l = l.strip('\n')
            items = [int(i) for i in l.split(' ')]
            iid, cate_ids = items[0], items[1:]
            iidcate_map[iid] = cate_ids
    return iidcate_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_root', type=str, default='datasets/beauty/train_3.txt')
    parser.add_argument('--test_root', type=str, default='datasets/beauty/test_3.txt')
    parser.add_argument('--cateid_root', type=str, default='datasets/beauty/cate.txt')
    parser.add_argument('--l_kernel_emb', type=str, default='datasets/beauty/item_kernel_3.pkl')
    parser.add_argument('--cate_num', type=int, default=213)
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3, help="consistent with the postfix of dataset")
    # dpp arguments
    parser.add_argument('--neg_samples', type=int, default=3, help="Z")
    parser.add_argument('--dpp_loss', type=int, default=2, help="0:cross-entropy, 1:DSL, 2:CDSL")
    parser.add_argument('--batch_format', type=int, default=1, help="use minibatch format for dpp loss or not")
    # train arguments
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.001, help="[0.0005 0.001 0.0015], default 0.001") 
    parser.add_argument('--l2', type=float, default=1e-4)

    config = parser.parse_args()

    # model dependent arguments
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=50)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')
    model_parser.add_argument('--sigma_alpha', type=float, default=0.01)

    model_config = model_parser.parse_args()
    model_config.L = config.L

    # set seed for TensorFlow
    tf.random.set_seed(config.seed)

    # load dataset
    train = Interactions(config.train_root)
    train.to_sequence(config.L, config.T)

    test = Interactions(config.test_root,
                        user_map=train.user_map,
                        item_map=train.item_map)

    cate = get_cates_map(config.cateid_root)

    print(config)
    print(model_config)

    # fit model
    # Assuming Recommender class is now TensorFlow compatible
    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        learning_rate=config.learning_rate,
                        l2=config.l2,
                        neg_samples=config.neg_samples,
                        model_args=model_config)

    model.fit(train, test, cate, config, verbose=True)