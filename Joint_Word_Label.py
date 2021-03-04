import tensorflow as tf
from tensorflow.keras.layers import  Dense, Dropout
# from keras_multi_head import MultiHead, MultiHeadAttention
from tensorflow.keras.layers import LSTM, Dropout, Conv2D, Bidirectional, Lambda, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Dropout
from SelfAttention import MultiHeadAttention
from CNN3L import cnn_3_layer


def LEAM(query_word_rep, x_mask, W_class, coocurrance_matrix_, num_classes, input_y,name="prop_lbl"):
    # x_emb_0 = tf.squeeze(query_word_rep,) # b * s * e

    H = tf.keras.layers.Conv1D(300, 3, activation='relu', padding='SAME', input_shape=query_word_rep.shape)(query_word_rep)
    G_ = tf.keras.backend.dot(H, tf.compat.v1.transpose(W_class, [1, 0]))  # b * s * c
    Att_v_max_ =  partial_softmax(G_, x_mask, 1, name + 'Att_v_max_l1') # b * s * c  # b * s * c
    x_att_ = tf.keras.backend.batch_dot(tf.compat.v1.transpose(H, [0, 2, 1]), Att_v_max_)
    H_enc_ = tf.unstack(x_att_, axis=-1)
    logits_list_ = []
    for i, ih in enumerate(H_enc_):
        score, l2_loss = discriminator_0layer(ih, name=name+ str(i))
        logits_list_.append(score)

    logits_ = tf.concat(logits_list_, -1)
    probs = tf.compat.v1.nn.softmax(logits_)
    losses = tf.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=input_y, logits=logits_))


    return probs, losses





def proposed_method_CM(query_word_rep, x_mask, W_class, coocurrance_matrix_, num_classes, input_y, name="prop_lbl"):
    """ compute the average over all word embeddings """

    W_class_tran = tf.compat.v1.transpose(W_class, [1, 0])
    H = tf.keras.layers.Conv1D(100, 3, activation='relu', padding='SAME', input_shape=query_word_rep.shape)(query_word_rep)
    G = tf.keras.backend.dot(H, tf.transpose(W_class, [1,0]))  # b * s * c
    att_words =  partial_softmax(G, x_mask, 1, name + 'Att_v_max_l1') # b * s * c  # b * s * c
    # att_words = MultiHeadAttention(head_num=num_classes, name= 'Multi-Head', )(G)
    # temp_mha = MultiHeadAttention(d_model= num_classes, num_heads=num_classes)
    # att_words, attn = temp_mha(G, k=G, q=G, mask=None)
    x_att1 = tf.keras.backend.batch_dot(tf.transpose(H, [0, 2, 1]), att_words)
    H_enc_ = tf.unstack(x_att1, axis=-1)
    logits_list_ = []
    for i, ih in enumerate(H_enc_):
        score, l2_loss = discriminator_0layer(ih, name=name + str(i))
        logits_list_.append(score)

    logits_ = tf.concat(logits_list_, -1)


    vt_v = tf.matmul(W_class, W_class_tran)
    sigmoid_x_att_ = tf.nn.sigmoid(vt_v)
    loss_within_LINE = tf.multiply(-1.0, tf.reduce_sum(tf.multiply(tf.compat.v1.log(sigmoid_x_att_), coocurrance_matrix_)))


    return logits_, loss_within_LINE



def proposed_method_TM(W_1, W_2, transition_matrix, name="prop_lbl"):

    trans_1_2 = tf.keras.backend.dot(W_1, tf.compat.v1.transpose(W_2, [1, 0]))
    softmax_trans_1_2 = tf.compat.v1.nn.softmax(trans_1_2, axis=1)
    loss_between = tf.multiply(-1.0, tf.reduce_sum(tf.multiply(tf.compat.v1.log(softmax_trans_1_2), transition_matrix)))


    return loss_between




def discriminator_0layer(H, name):
    with tf.compat.v1.name_scope(name+"output_"):
        W = tf.compat.v1.get_variable(
            "W_" + name,
            shape=[H.get_shape()[1], 1],
            initializer=tf.initializers.glorot_normal())
        b = tf.Variable(tf.constant(0.1, shape=[1]), name="b_" + name)
        l2_loss = tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        scores = tf.compat.v1.nn.xw_plus_b(H, W, b, name=name+"scores_")
    return scores, l2_loss

def discriminator_2layer(H, output_size, name):
    # Final (unnormalized) scores and predictions
    fc1 = Dense(256, activation='relu', name=name + 'fc1')(H)
    fc1 = Dropout(0.3)(fc1)
    fc2 = Dense(128, activation='relu', name=name + 'fc2')(fc1)
    fc2 = Dropout(0.3)(fc2)
    scores = Dense(output_size, activation='sigmoid', name=name + 'fc3')(fc2)
    return scores


def discriminator_1layer(H, output_size, name):
    # Final (unnormalized) scores and predictions
    scores = Dense(output_size, activation='sigmoid', name=name + 'fc1')(H)
    return scores


def partial_softmax(logits, weights, dim, name,):
    with tf.compat.v1.name_scope(name + 'partial_softmax'):
        exp_logits = tf.exp(logits)
        if len(exp_logits.get_shape()) == len(weights.get_shape()):
            exp_logits_weighted = tf.multiply(exp_logits, weights)
        else:
            exp_logits_weighted = tf.multiply(exp_logits, tf.compat.v1.expand_dims(weights, -1))
        exp_logits_sum = tf.math.reduce_sum(exp_logits_weighted, axis=dim, keepdims = True)
        partial_softmax_score = tf.math.divide(exp_logits_weighted, exp_logits_sum, name=name)
        return partial_softmax_score




