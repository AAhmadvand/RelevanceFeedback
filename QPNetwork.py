import tensorflow as tf
import Mix_encoder_product
import QP2Vec
import Joint_Word_Label
from CNN3L import cnn_3_layer,Bidirectional_LSTM
from SelfAttention import MultiHeadAttention
from tensorflow.keras.layers import LSTM, Dropout, Conv2D, Bidirectional, Lambda, GlobalAveragePooling1D, GlobalAveragePooling2D, Dense, Dropout, Flatten, Activation, RepeatVector, Permute, Attention

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


class QPNetwork(object):
    """
    
    QP2Vector for query category mapping.
    
    """
    def __init__(self, sequence_length, sequence_length_prod, num_classes_L6, num_quantized_chars, word_char_length, vocab_size,
                 embedding_size, embedding_char_size, filter_sizes, num_filters, top_K, l2_reg_lambda=0.0, Ranking = False, K = 1, only_query= True):

            # Placeholders for input, output and dropout
            self.input_x_query = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], "input_x_query")
            self.input_x_prod = tf.compat.v1.placeholder(tf.int32, [None, 4*top_K, sequence_length_prod], "input_x_prod")

            self.input_char_query = tf.compat.v1.placeholder(tf.int32, [None, sequence_length, word_char_length],"input_char_query")
            self.input_char_prod = tf.compat.v1.placeholder(tf.int32, [None, 4*top_K, sequence_length_prod, word_char_length],"input_char_prod")

            self.input_x_mask = tf.compat.v1.placeholder(tf.float32, [None, sequence_length], name="input_x")


            self.input_indices_L6 = tf.compat.v1.placeholder(tf.int64, [None, 2], name="input_indices_L6")
            self.input_values_L6 = tf.compat.v1.placeholder(tf.float32, [None], name="input_values_L6")


            self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
            self.size_sample = tf.compat.v1.placeholder(tf.float32, name="size_input")
            
            
            self.coocurrance_matrix = tf.compat.v1.placeholder(tf.float32, [num_classes_L6, num_classes_L6], name="co_mat_l6")


            
#             for i in range(top_K/4):

#             self.title, self.desc = tf.split(self.input_x_prod, [sequence_length, sequence_length], 1)
#             self.title_char, self.desc_char = tf.split(self.input_char_prod, [sequence_length, sequence_length], 1)

            
            self.input_x_query_ = tf.expand_dims(self.input_x_query, axis=1)
            self.input_x_query_prod = tf.concat([self.input_x_query_, self.input_x_prod], axis=1)
            

            self.input_char_query_ = tf.expand_dims(self.input_char_query, axis=1)
            self.input_char_query_prod = tf.concat([self.input_char_query_, self.input_char_prod], axis=1)

            # Keeping track of l2 regularization loss (optional)
            l2_loss_L2 = tf.constant(l2_reg_lambda)
            l2_loss_L1 = tf.constant(l2_reg_lambda)
            l2_loss_L6 = tf.constant(l2_reg_lambda)

            ############################################################################################################
            # Embedding layer
            with tf.compat.v1.name_scope("embedding_prod"):
                self.W = tf.Variable(tf.compat.v1.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_prod", trainable=True)
                self.query_product_embedded_words = tf.compat.v1.nn.embedding_lookup(self.W, self.input_x_query_prod)

            ############################################################################################################
            # Embedding layer
            with tf.compat.v1.name_scope("embedding_char_prod"):
                W_char = tf.Variable(tf.compat.v1.random_uniform([num_quantized_chars, embedding_char_size], -1.0, 1.0),name="W_char_prod", trainable=True)
                embedded_chars_ = tf.compat.v1.nn.embedding_lookup(W_char, self.input_char_query_prod)
                self.query_product_embedded_chars = tf.compat.v1.transpose(embedded_chars_, [1, 2, 0, 3, 4])
            ############################################################################################################
            self.query_word_rep, self.product_word_rep = Mix_encoder_product.Mix_encoder_product(sequence_length,
                                                                                                 word_char_length,
                                                                                                 vocab_size, embedding_size,
                                                                                                 embedding_char_size,
                                                                                                 filter_sizes, num_filters,
                                                                                                 self.query_product_embedded_words,
                                                                                                 self.query_product_embedded_chars,
                                                                                                 l2_reg_lambda=0.0,
                                                                                                 top_K=top_K * 4 + 1, name_qp="mix")

            print(len(self.product_word_rep))
            indice_p = [0]
            self.query_embedded_words_ = tf.compat.v1.gather(self.query_product_embedded_words, indice_p, axis=1)
            self.query_embedded_words_ = tf.compat.v1.transpose(self.query_embedded_words_, [0, 2, 3, 1])
            self.query_embedded_words = tf.reduce_mean(self.query_embedded_words_, axis=3)

            self.query_vector = QP2Vec.QP2Vec(sequence_length, filter_sizes, num_filters, self.query_word_rep, name_qp="query")
#             self.query_vector = cnn_3_layer(self.query_embedded_words, sequence_length, embedding_size, name="cnn3l")
            # self.query_vector = GlobalAveragePooling1D()(self.query_embedded_words)
#             print("self.query_vector", self.query_vector.shape)
            
        
            qp_vectors = []
            for product_id in range(top_K*4):
                print("i: ", product_id)
                indice_p = [product_id + 1]
                self.prod_embedded_words_ = tf.compat.v1.gather(self.query_product_embedded_words, indice_p, axis=1)
                self.prod_embedded_words_ = tf.compat.v1.transpose(self.prod_embedded_words_, [0, 2, 3, 1])
                self.product_embedded_words = tf.reduce_mean(self.prod_embedded_words_, axis=3)
                

#                 CNN_3 = cnn_3_layer(self.product_embedded_words, sequence_length, embedding_size, name="prod_cnn3l" + str(product_id))
#                 qp_vectors.append(CNN_3)
#                 qp_vectors.append(QP2Vec.QP2Vec(sequence_length, filter_sizes, num_filters, self.product_word_rep[product_id], name_qp="product"+str(product_id)))
#                 QP2Vec.QP2Vec(sequence_length, filter_sizes, num_filters, self.product_word_rep[product_id],name_qp="product" + str(product_id))         
#                 qp_vectors.append(GlobalAveragePooling1D()(self.product_word_rep[product_id]))
#                 qp_vectors.append(tf.reduce_mean(self.product_word_rep, axis=1))

                     
#             print("CNN_3.shape[-1]", CNN_3.shape[-1])
#             self.query_vector = tf.keras.layers.Dense(CNN_3.shape[-1])(self.query_vector)

            query_seq_encoding = tf.transpose(tf.compat.v1.expand_dims(self.query_vector, -1),[0,2,1])
            query_encoding = tf.keras.layers.GlobalAveragePooling1D()(tf.compat.v1.nn.l2_normalize(query_seq_encoding, -1))
            
#             updated_qp_vectors = []
#             for product in qp_vectors:
#                 # normalize input
#                 norm_product = tf.compat.v1.expand_dims(tf.compat.v1.nn.l2_normalize(product, -1), -1)
#                 norm_query = tf.compat.v1.expand_dims(tf.compat.v1.nn.l2_normalize(self.query_vector, -1), -1)
#                 similarity = tf.reduce_mean(tf.matmul(tf.compat.v1.transpose(norm_query, [0, 2, 1]), norm_product),axis=2)
#                 updated_qp_vectors.append(tf.multiply(similarity, product))


#             ########if there is not any Product desc added to the vector

#             if len(updated_qp_vectors) != 0:
#                 new_product_vectors = tf.stack(updated_qp_vectors, -1)
#                 new_product_vectors = tf.transpose(new_product_vectors, [0,2,1])
#                 temp_mha = MultiHeadAttention(d_model= 420, num_heads = len(updated_qp_vectors))
#                 att_words, attn = temp_mha(new_product_vectors, k=new_product_vectors, q=new_product_vectors, mask=None)
#                 self.product_vector = tf.reduce_mean(att_words, axis=1)
            
            
            updated_qp_vectors = []
            for product in qp_vectors:
                product = tf.transpose(tf.compat.v1.expand_dims(tf.compat.v1.nn.l2_normalize(product, -1), -1),[0,2,1])
#                 print ("product", product.shape)
                query_attention_fields = tf.keras.layers.Attention()([query_seq_encoding, product])   
#                 print("query_attention_fields", query_attention_fields.shape)
                # Reduce over the sequence axis to produce encodings of shape
                query_value_attention_fields = tf.keras.layers.GlobalAveragePooling1D()(query_attention_fields)
                print ("query_value_attention_fields", query_value_attention_fields.shape)
                # Concatenate query and document encodings to produce a DNN input layer.
    #           query_value_attention_fields = tf.keras.layers.Concatenate()([query_value_attention_fields])
                updated_qp_vectors.append(query_value_attention_fields)
        
        
        
#             updated_qp_vectors_docs = []
#             for i in range(top_K):
#                 doc = tf.transpose(tf.compat.v1.nn.l2_normalize(updated_qp_vectors[4*i:4*(i+1)], -1),[1,0,2])
#                 print ("doc", doc.shape)
#                 query_attention_fields_doc = tf.keras.layers.Attention()([query_seq_encoding, doc])   
# #                 print("query_attention_fields", query_attention_fields.shape)
#                 # Reduce over the sequence axis to produce encodings of shape
#                 query_value_attention_doc = tf.keras.layers.GlobalAveragePooling1D()(query_attention_fields_doc)
#                 # Concatenate query and document encodings to produce a DNN input layer.
#     #           query_value_attention_fields = tf.keras.layers.Concatenate()([query_value_attention_fields])
#                 print ("query_value_attention_doc", query_value_attention_doc.shape)

#                 updated_qp_vectors_docs.append(query_value_attention_doc)

#             print()
            if len(updated_qp_vectors) != 0:
                new_product_vectors = tf.stack(updated_qp_vectors, -1)
                new_product_vectors = tf.transpose(new_product_vectors, [0,2,1])
                temp_mha = MultiHeadAttention(d_model= len(updated_qp_vectors) * 10, num_heads = len(updated_qp_vectors))
                att_words, attn = temp_mha(new_product_vectors, k=new_product_vectors, q=new_product_vectors, mask=None)
                self.product_vector = tf.reduce_mean(att_words, axis=1)

            
#             doc_att_stack = tf.transpose(tf.stack(doc_stack, -1),[0,2,1])
#             print("doc_att_stack", doc_att_stack.shape)      
#             ########if there is not any Product desc added to the vector                
#             # Query-value attention of shape [batch_size, Tq, filters].
#             query_attentions_docs = tf.keras.layers.Attention()([query_seq_encoding, doc_att_stack])
#             # Reduce over the sequence axis to produce encodings of shape
#             query_value_attention_docs = tf.keras.layers.GlobalAveragePooling1D()(query_attentions_docs)
#             # Concatenate query and document encodings to produce a DNN input layer.
#             qp_vectors_pseudo = tf.keras.layers.Concatenate()([query_encoding, query_value_attention_docs])




            ############################################################################################################
#             doc_stack = []
#             for i in range(top_K):
#                 doc_rep = tf.transpose(tf.stack(qp_vectors[i*4:(i+1)*4], -1),[0,2,1])            
#                 temp_mha = MultiHeadAttention(d_model=128, num_heads=8)
#                 att_words, attn = temp_mha(doc_rep, k=doc_rep, q=doc_rep, mask=None)
#                 query_value_attention = GlobalAveragePooling1D()(att_words)
#                 doc_stack.append(query_value_attention)
         
        
            ############### Query _ Product Vector

            if not only_query:
                self.query_product_vector = tf.keras.layers.Concatenate()([self.query_vector, self.product_vector])
            else:
                self.query_product_vector = self.query_vector
    

            ############################################################################################################
            ############################################################################################################
            
             # Embedding layer For Labels
            with tf.compat.v1.device('/cpu:0'), tf.compat.v1.name_scope("embedding_L6"):
                self.W_L6 = tf.Variable(tf.compat.v1.random_uniform([num_classes_L6, self.query_embedded_words.shape[2]], -0.001, 0.001), name="W_l6", trainable=True)
            print (self.W_L6.shape)
            
            ############################################################################################################
            ############################################################################################################
            
            with tf.compat.v1.name_scope("REST"):
            
                self.input_y_L6_ = tf.sparse.SparseTensor(indices=self.input_indices_L6, values=self.input_values_L6, dense_shape=[self.size_sample, num_classes_L6])
                self.input_y_L6 = tf.sparse.to_dense(self.input_y_L6_, default_value=None, validate_indices=True, name="input_L6")


                if not Ranking :
                    comapre = 1e-7 * tf.ones([self.size_sample, self.input_y_L6.shape[1]], dtype='float32')
                    self.input_y_L6 = tf.cast(tf.math.greater(self.input_y_L6, comapre), dtype='float32')


                self.query_product_vector_L6 = tf.concat([self.query_product_vector], 1)

                ###########################################################################################################
                ###########################################################################################################
                
#                 self.probs_L6, self.loss_L6 = Joint_Word_Label.LEAM(self.query_embedded_words,
#                                                                                  self.input_x_mask,
#                                                                                  self.W_L6,
#                                                                                  self.coocurrance_matrix,
#                                                                                  num_classes_L6,
#                                                                                  self.input_y_L6,
#                                                                                  name='lbl_L6')
                

                self.probs_L6, self.loss_L6 = QP2Vec.compute_loss(self.query_product_vector_L6, self.input_y_L6, self.dropout_keep_prob, num_classes_L6 , name = "L6")

                ###########################################################################################################
                ###########################################################################################################
                
                            
                values_all_1, self.indices_all_1 = tf.math.top_k(self.input_y_L6, k=10, sorted=True, name=None)
                values_true_1, self.indices_true_1= tf.math.top_k(self.input_y_L6, k=1, sorted=True, name=None)
                values_pred_1, self.indices_pred_1 = tf.math.top_k(self.probs_L6, k=1, sorted=True, name=None)

                self.all_relevant_1 = tf.cast(tf.greater(values_all_1, tf.constant(1e-7)), tf.float32)
                self.true_relevant_K_1 = tf.cast(tf.greater(values_true_1, tf.constant(1e-7)), tf.float32)
                self.pred_relevant_K_1 = tf.cast(tf.greater(values_pred_1, tf.constant(1 / (4))), tf.float32)

                self.all_relevant_indices_1 = tf.math.multiply(self.all_relevant_1, tf.cast(self.indices_all_1, tf.float32))
                self.true_relevnt_indices_K_1 = tf.math.multiply(self.true_relevant_K_1, tf.cast(self.indices_true_1, tf.float32))
                self.pred_relevnat_indices_K_1 = tf.math.multiply(self.pred_relevant_K_1,tf.cast(self.indices_pred_1, tf.float32))

                self.num_relevant_items_1 = tf.math.reduce_sum(self.all_relevant_1, axis=1)
                self.num_recom_items_k_1 = tf.math.reduce_sum(self.pred_relevant_K_1, axis=1)

                condition_1 = tf.equal(self.pred_relevnat_indices_K_1, 0)
                case_true_1 = tf.cast(tf.reshape(tf.multiply(tf.ones([self.size_sample * 1], tf.int32), -9999), [self.size_sample, 1]),tf.float32)
                case_false_1 = self.pred_relevnat_indices_K_1
                self.pred_relevnat_indices_K_replace_1 = tf.where(condition_1, case_true_1, case_false_1)

                condition_1 = tf.equal(self.all_relevant_indices_1, 0)
                case_true_1 = tf.cast(tf.reshape(tf.multiply(tf.ones([self.size_sample * 10], tf.int32), -88888), [self.size_sample, 10]),tf.float32)
                case_false_1 = self.all_relevant_indices_1
                self.all_relevant_indices_replace_1 = tf.where(condition_1, case_true_1, case_false_1)

                ###########################################################################################################
                ###########################################################################################################
                values_all_2, self.indices_all_2 = tf.math.top_k(self.input_y_L6, k=10, sorted=True, name=None)
                values_true_2, self.indices_true_2 = tf.math.top_k(self.input_y_L6, k=2, sorted=True, name=None)
                values_pred_2, self.indices_pred_2 = tf.math.top_k(self.probs_L6, k=2, sorted=True, name=None)

                self.all_relevant_2 = tf.cast(tf.greater(values_all_2, tf.constant(1e-7)), tf.float32)
                self.true_relevant_K_2 = tf.cast(tf.greater(values_true_2, tf.constant(1e-7)), tf.float32)
                self.pred_relevant_K_2 = tf.cast(tf.greater(values_pred_2, tf.constant(1 / (8))), tf.float32)

                self.all_relevant_indices_2 = tf.math.multiply(self.all_relevant_2, tf.cast(self.indices_all_2, tf.float32))
                self.true_relevnt_indices_K_2 = tf.math.multiply(self.true_relevant_K_2, tf.cast(self.indices_true_2, tf.float32))
                self.pred_relevnat_indices_K_2 = tf.math.multiply(self.pred_relevant_K_2,tf.cast(self.indices_pred_2, tf.float32))

                self.num_relevant_items_2 = tf.math.reduce_sum(self.all_relevant_2, axis=1)
                self.num_recom_items_k_2 = tf.math.reduce_sum(self.pred_relevant_K_2, axis=1)

                condition_2 = tf.equal(self.pred_relevnat_indices_K_2, 0)
                case_true_2 = tf.cast(tf.reshape(tf.multiply(tf.ones([self.size_sample * 2], tf.int32), -9999), [self.size_sample, 2]),tf.float32)
                case_false_2 = self.pred_relevnat_indices_K_2
                self.pred_relevnat_indices_K_replace_2 = tf.where(condition_2, case_true_2, case_false_2)

                condition_2 = tf.equal(self.all_relevant_indices_2, 0)
                case_true_2 = tf.cast(tf.reshape(tf.multiply(tf.ones([self.size_sample * 10], tf.int32), -88888), [self.size_sample, 10]),tf.float32)
                case_false_2 = self.all_relevant_indices_2
                self.all_relevant_indices_replace_2 = tf.where(condition_2, case_true_2, case_false_2)
                
                ###########################################################################################################
                ###########################################################################################################
                
                values_all_3, self.indices_all_3 = tf.math.top_k(self.input_y_L6, k=10, sorted=True, name=None)
                values_true_3, self.indices_true_3 = tf.math.top_k(self.input_y_L6, k=3, sorted=True, name=None)
                values_pred_3, self.indices_pred_3 = tf.math.top_k(self.probs_L6, k=3, sorted=True, name=None)

                self.all_relevant_3 = tf.cast(tf.greater(values_all_3, tf.constant(1e-7)), tf.float32)
                self.true_relevant_K_3 = tf.cast(tf.greater(values_true_3, tf.constant(1e-7)), tf.float32)
                self.pred_relevant_K_3 = tf.cast(tf.greater(values_pred_3, tf.constant(1 / (16))), tf.float32)

                self.all_relevant_indices_3 = tf.math.multiply(self.all_relevant_3, tf.cast(self.indices_all_3, tf.float32))
                self.true_relevnt_indices_K_3 = tf.math.multiply(self.true_relevant_K_3, tf.cast(self.indices_true_3, tf.float32))
                self.pred_relevnat_indices_K_3 = tf.math.multiply(self.pred_relevant_K_3,tf.cast(self.indices_pred_3, tf.float32))

                self.num_relevant_items_3 = tf.math.reduce_sum(self.all_relevant_3, axis=1)
                self.num_recom_items_k_3 = tf.math.reduce_sum(self.pred_relevant_K_3, axis=1)

                condition_3 = tf.equal(self.pred_relevnat_indices_K_3, 0)
                case_true_3 = tf.cast(tf.reshape(tf.multiply(tf.ones([self.size_sample * 3], tf.int32), -9999), [self.size_sample, 3]),tf.float32)
                case_false_3 = self.pred_relevnat_indices_K_3
                self.pred_relevnat_indices_K_replace_3 = tf.where(condition_3, case_true_3, case_false_3)

                condition_3 = tf.equal(self.all_relevant_indices_3, 0)
                case_true_3 = tf.cast(tf.reshape(tf.multiply(tf.ones([self.size_sample * 10], tf.int32), -88888), [self.size_sample, 10]),tf.float32)
                case_false_3 = self.all_relevant_indices_3
                self.all_relevant_indices_replace_3 = tf.where(condition_3, case_true_3, case_false_3)

                ###################################################################################
                ###################################################################################
                # overall loss
                with tf.compat.v1.name_scope("overall_loss"):
                    self.loss = self.loss_L6
