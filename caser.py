import tensorflow as tf
from tensorflow.keras import layers, Model

from utils import activation_getter

class Caser(Model):
    def __init__(self, num_users, num_items, model_args):
        super(Caser, self).__init__()
        self.args = model_args

        # init args
        L = self.args.L
        dims = self.args.d
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]

        # user and item embeddings
        self.user_embeddings = layers.Embedding(num_users, dims)
        self.item_embeddings = layers.Embedding(num_items, dims)

        # vertical conv layer
        self.conv_v = layers.Conv2D(self.n_v, (L, 1), padding='same')

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = [layers.Conv2D(self.n_h, (i, dims), padding='same') for i in lengths]

        # fully-connected layer
        self.fc1_dim_v = 5 * 1 * self.n_v
        self.fc1_dim_h = self.n_h * len(lengths)
        self.fc1 = layers.Dense(dims)

        # W2, b2 embeddings
        self.W2 = layers.Embedding(num_items, dims+dims)
        self.b2 = layers.Embedding(num_items, 1)

        # dropout
        self.dropout = layers.Dropout(self.drop_ratio)

    
    def call(self, seq_var, user_var, item_var, for_pred=False):
        item_embs = tf.expand_dims(self.item_embeddings(seq_var), 1)
        user_emb = tf.squeeze(self.user_embeddings(user_var), 1)

        out, out_h, out_v = None, None, None
        if self.n_v:
            out_v = self.conv_v(item_embs)
            print("Shape of out_v after convolution:", out_v.shape)

            # Dynamically compute self.fc1_dim_v based on the output shape of the vertical convolution
            _, height, width, channels = out_v.shape
            self.fc1_dim_v = height * width * channels

            # Compute the expected number of values for reshaping
            batch_size = tf.shape(seq_var)[0]
            expected_values = batch_size * self.fc1_dim_v

            if expected_values != self.fc1_dim_v * batch_size:
                raise ValueError(f"Expected {self.fc1_dim_v * batch_size} values for reshaping, but got {expected_values} values.")

            out_v = tf.reshape(out_v, [-1, self.fc1_dim_v])
            print("Shape of out_v after reshape:", out_v.shape)
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs))
                pool_out = tf.keras.layers.MaxPooling1D(pool_size=(tf.shape(conv_out)[2],))(tf.squeeze(conv_out, axis=1))
                out_hs.append(pool_out)
            out_h = tf.concat(out_hs, axis=1)

        # Reshape out_h to ensure it has the same rank as out_v
        out_h = tf.reshape(out_h, [-1, self.n_h * len(self.conv_h)])

        out = tf.concat([out_v, out_h], axis=1)
        out = self.dropout(out)

        z = self.ac_fc(self.fc1(out))
        x = tf.concat([z, user_emb], 1)

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = tf.squeeze(w2)
            b2 = tf.squeeze(b2)
            res = tf.reduce_sum(x * w2, axis=1) + b2
        else:
            res = tf.squeeze(tf.linalg.matmul(w2, tf.expand_dims(x, 2)) + b2)

        return res
