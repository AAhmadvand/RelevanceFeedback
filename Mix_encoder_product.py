import tensorflow as tf


def Mix_encoder_product( sequence_length, word_char_length, vocab_size,
                 embedding_size, embedding_char_size, filter_sizes, num_filters,  embedded_words, embedded_chars,
                 l2_reg_lambda=0.0, top_K = 2, name_qp="mix"):


        
        print("%%%%%%%%%%%%%%%%%%%%")
        print("%%%%%%%%%%%%%%%%%%%%")
        print(top_K)

        all_logits_list = []
        for k in range(top_K):

            indice_p = [k]
            char_vector_p = tf.compat.v1.gather(embedded_chars, indice_p, axis=0)

            logits_list = []
            for j in range(sequence_length):
                # print(j)
                indice = [j]

                char_vector_ = tf.compat.v1.gather(char_vector_p, indice, axis=1)
                char_vector1 = tf.compat.v1.transpose(char_vector_, [2, 3, 4, 1, 0])
                char_vector = tf.reduce_sum(char_vector1, axis=4)

                num_filters_ch = num_filters
                pooled_ch_outputs = []
                k_pooled_ch_outputs = []
                for i, filter_size in enumerate(filter_sizes):
                    with tf.compat.v1.name_scope(name_qp + "conv-maxpool-word-%s" % filter_size):
                        # Convolution Layer
                        filter_shape = [filter_size, char_vector.shape[2], 1, num_filters_ch]
                        W = tf.Variable(tf.compat.v1.truncated_normal(filter_shape, stddev=0.1), name=name_qp + "W_w")
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters_ch]), name=name_qp + "b_w")
                        conv = tf.compat.v1.nn.conv2d(
                            char_vector,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name=name_qp + "conv_w")

                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name=name_qp + "relu_w")
                        # Maxpooling over the outputs

                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, word_char_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name=name_qp + "pool_w")

                        ####  K-Max Pooling
                        # size_ = word_char_length - filter_size + 1
                        # kmax_pool_ch_cnn = KMaxPooling(h, size_, num_filters_ch, filter_sizes, name_qp)

                        pooled_ch_outputs.append(pooled)
                        k_pooled_ch_outputs.append(pooled)

                # Combine all the pooled features
                num_filters_total = num_filters * len(filter_sizes)
                h_pool = tf.concat(pooled_ch_outputs, 3)
                h_pool_kmax_ch_cnn = tf.reshape(h_pool, [-1, num_filters_total])
                logits_list.append(h_pool_kmax_ch_cnn)

            all_logits_list.append(logits_list)

        product_word_representation = []
        query_word_representation = None
        for in_it, logits_list in enumerate(all_logits_list):


            embedded_chars = tf.stack(logits_list, -1)
            embedding_char_wordlevel_emb = tf.compat.v1.transpose(embedded_chars, [0, 2, 1])

            ################################# HighWay ##########################################
            ####################################################################################

            # b_hc = tf.Variable(tf.constant(0.1, shape=[embedding_char_wordlevel_emb.shape[-1]]), name=name_qp + "b_hc")
            # highway_hc_ = tf.nn.relu(tf.nn.bias_add(embedding_char_wordlevel_emb, b_hc), name=name_qp + "relu_hc")
            # highway_hc = tf.compat.v1.nn.sigmoid(highway_hc_)

            ###############################Concat Word + Char Embedding#########################
            ####################################################################################

            indice_w = [in_it]
            wor_r = tf.compat.v1.gather(embedded_words, indice_w, axis=1)
            wor_r1 = tf.compat.v1.transpose(wor_r, [0, 2, 3, 1])
            highway_hw = tf.reduce_sum(wor_r1, axis=3)
            ####################################################################################

            word_rep_ = tf.concat([highway_hw, embedding_char_wordlevel_emb], -1)
            if in_it == 0:
                query_word_representation = word_rep_
            else:
                product_word_representation.append(word_rep_)



        return query_word_representation, product_word_representation


