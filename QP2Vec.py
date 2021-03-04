import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LSTM, Dropout, Conv2D, Bidirectional, Lambda, GlobalAveragePooling1D, GlobalAveragePooling2D, Dense, Dropout
# import keras
# from keras_self_attention import SeqSelfAttention
from SelfAttention import MultiHeadAttention



def QP2Vec(sequence_length, filter_sizes, num_filters, ME_word_rep, name_qp = "qp"):

        word_rep_expanded = tf.compat.v1.expand_dims(ME_word_rep, -1)

        ####################################################################################
        ####################################################################################

        pooled_outputs = []
        k_pooled_word_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.compat.v1.name_scope(name_qp+"conv-maxpool-word-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, word_rep_expanded.shape[2], 1, num_filters]
                W = tf.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name=name_qp+"W_w")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name=name_qp+"b_w")
                conv = tf.compat.v1.nn.conv2d(
                    word_rep_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name=name_qp+"conv_w")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name=name_qp+"relu_w")

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name=name_qp+"pool_w")

                k_pooled_word_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(k_pooled_word_outputs, 3)
        h_pool_kmax_word_cnn = tf.reshape(h_pool, [-1, num_filters_total])

        ######################### Multi-Head SELF-ATTENTION Word-Level######################
        ####################################################################################

        temp_mha = MultiHeadAttention(d_model=200, num_heads=10)
        att_words, attn = temp_mha(ME_word_rep, k=ME_word_rep, q=ME_word_rep, mask=None)
        self_words_att = GlobalAveragePooling1D()(att_words)

        ####################################################################################
        ####################################################################################

        h_pool2_flat = tf.concat([self_words_att , h_pool_kmax_word_cnn], 1)

        ##############################Average Word Embedding################################
        ##############################Average Word Embedding################################

        average_embedding = tf.reduce_mean(ME_word_rep, axis=1)
        h_pool2_flat = tf.concat([h_pool2_flat, average_embedding], 1)
#         bias_last = tf.Variable(tf.constant(0.1, shape=[h_pool2_flat.shape[1]]), name=name_qp+"b")
#         relu_cat_last = tf.nn.relu(tf.nn.bias_add(h_pool2_flat, bias_last), name=name_qp+"relu")
        # qp_q = tf.compat.v1.nn.tanh(relu_cat_last)
        qp_q =  h_pool2_flat

        return qp_q



def compute_loss(query_product_vector, input_y, dropout_keep_prob,  num_classes, name = "loss_"):

    # Add dropout
    with tf.compat.v1.name_scope(name + "dropout"):
        h_drop = tf.compat.v1.nn.dropout(query_product_vector, rate=1 - dropout_keep_prob)

    # Final (unnormalized) scores and predictions
    with tf.compat.v1.name_scope(name + "output"):
        W = tf.compat.v1.get_variable(
            name + "W",
            shape=[query_product_vector.get_shape()[1], num_classes],
            initializer=tf.initializers.glorot_normal())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name= name + "b")
        scores = tf.compat.v1.nn.xw_plus_b(h_drop, W, b, name= name + "scores")

    # CalculateMean cross-entropy loss
    with tf.compat.v1.name_scope(name + "loss_"):
        probs = tf.compat.v1.nn.softmax(scores)
        losses = tf.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=input_y, logits=scores))


    return probs, losses