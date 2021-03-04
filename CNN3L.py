import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Conv2D, Bidirectional, Lambda, GlobalAveragePooling1D, GlobalAveragePooling2D, Dense, Dropout, Flatten, Activation, RepeatVector, Permute, Attention


def cnn_3_layer(x_emb_0, maxlen, embedding_size, name):

    # x_emb_0 = tf.squeeze(x_emb, )  # b * s * e
    x_emb_0 = tf.expand_dims(x_emb_0, -1)

    num_filters = 128
    filter_sizes = [1, 2, 3]

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope(name+"conv-maxpool-%s"% filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                x_emb_0,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv"+ name)

#             conv = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, center=True, scale=True)(conv)

            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu"+ name)
            # Maxpooling over the outputs
            
            
             ####  K-Max Pooling
#             size_ = maxlen - filter_size + 1
#             kmax_poolin = KMaxPooling(h, size_, num_filters, filter_sizes, name_qp= "kmax"+ name)
#             print (kmax_poolin.shape)
#             print("%%%%%%%%%%%%%%%%%")
            
            
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, maxlen - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool"+ name)
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 1)
#     print (h_pool.shape)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
#     print (h_pool_flat.shape)
#     print("*****")

    return h_pool_flat


def KMaxPooling(input_val, size_, num_filters, filter_sizes, name_qp = "mix"):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    lower = size_ // 3
    middle = lower
    higher = size_ - 2 * middle
    sub_w = tf.split(input_val, num_or_size_splits=[lower, middle, higher], axis=1)

    pooled_outputs = []
    for sub in sub_w:
        pooled = tf.nn.max_pool(
            sub,
            ksize=[1, sub.shape[1], 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name=name_qp+"pool_kmax")

        pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_cnn = tf.reshape(h_pool, [-1, num_filters_total])
#     h_pool_cnn = tf.compat.v1.expand_dims(tf.compat.v1.expand_dims(h_pool_cnn, 1),1 )

    return h_pool_cnn



def Bidirectional_LSTM(x_emb_0, maxlen, embedding_size, name):
    
    layer = Bidirectional(LSTM(256))(x_emb_0)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
#     layer = Dense(128,name='out_layer')(layer)
    layer2 = tf.keras.layers.Attention()([layer, layer]) 
    layer = tf.keras.layers.Concatenate()([layer, layer2])
    
    return layer