import tensorflow as tf
import numpy as np
import os, warnings, sys
import time , collections
import datetime
import data_helpers
import pickle
import numpy as np
from QP2Vec import QP2Vec
# from sklearn.metrics import ndcg_score
# from tensorflow.contrib import learn
# from sklearn.utils import shuffle
# from sklearn.preprocessing import normalize
from QPNetwork import QPNetwork
import time, json
from embedding import Word2Vec
from sklearn.metrics import classification_report as cr
from evaluation import macro_f1, micro_f1, precision_at_k, ndcg_score, fast_precision_recall_at_K
from sklearn.metrics import f1_score


print(tf.__version__)
# Parameters
# ==================================================
#ss


try:
    os.remove("result.txt")
except:
    pass

try:
    os.remove("result_multi_label.txt")
except:
    pass

# Data loading params
tf.compat.v1.flags.DEFINE_float("dev_sample_percentage", 0.5, "Percentage of the training data to use for validation")
tf.compat.v1.flags.DEFINE_string("Training_Data","", "Training dataset")
tf.compat.v1.flags.DEFINE_string("Test_data_entity", "", "Test dataset")


# Model Hyperparameters
tf.compat.v1.flags.DEFINE_string("word2vec", "/home/jupyter/GoogleNews-vectors-negative300.bin", "Word2vec file with pre-trained embeddings (default: None)")
# tf.compat.v1.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings (default: None)")
tf.compat.v1.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of word embedding")
tf.compat.v1.flags.DEFINE_integer("embedding_char_dim", 16, "Dimensionality of entity embedding (default: 16)")
tf.compat.v1.flags.DEFINE_integer("num_quantized_chars", 49, "num_quantized_chars")
tf.compat.v1.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '1,2,3')")
tf.compat.v1.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.compat.v1.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.compat.v1.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.compat.v1.flags.DEFINE_integer("max_query_length", 10, "Maximum length of query (default: 8)")
tf.compat.v1.flags.DEFINE_integer("max_query_length_prod", 10, "Maximum length of documet fields (default: 2 * 8)")
tf.compat.v1.flags.DEFINE_integer("max_char_length", 10, "MAximum length of chars in query (default: 8)")
tf.compat.v1.flags.DEFINE_integer("top_K", 7, "Top Product description")
tf.compat.v1.flags.DEFINE_boolean("only_query", False, "Only uses query for everything")
tf.compat.v1.flags.DEFINE_boolean("Ranking", False, "Ranking? or MLC")
tf.compat.v1.flags.DEFINE_boolean("minorities", True, "Ranking? or MLC")
tf.compat.v1.flags.DEFINE_integer("K", 1, "P@K, R@K")
tf.compat.v1.flags.DEFINE_string("EXPANSION", 'seperate', "Type of Query Expansion")

# 2731
# Training parameters
tf.compat.v1.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.compat.v1.flags.DEFINE_integer("num_epochs", 6, "Number of training epochs (default: 200)")
tf.compat.v1.flags.DEFINE_integer("evaluate_every", 2731, "Evaluate model on dev set after this many steps (default: 100)")
tf.compat.v1.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.compat.v1.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.compat.v1.flags.FLAGS

# weighted_cross_entropy_with_logits

# Data Preparation
# ================================================== ==================================================

# Load data
print("Loading data...")
train_data_, train_data_char, train_label, test_data_,test_data_char, test_label, vocab, product_train, product_train_char, product_test, product_test_char, Bucket, w2v, minorities = data_helpers.load_data_and_labels(FLAGS.max_query_length_prod, FLAGS.max_query_length , FLAGS.max_char_length, FLAGS.top_K, FLAGS.EXPANSION, FLAGS.only_query, FLAGS.minorities)

vocab_processor = np.zeros([len(train_data_), FLAGS.max_query_length], dtype=int)
train_data  = data_helpers.fit_transform(train_data_, vocab_processor, vocab)
train_mask = np.float32(train_data > 0)

vocab_processor_prod = np.zeros([len(product_train), 4 * FLAGS.top_K, FLAGS.max_query_length_prod], dtype=int)
train_prod = data_helpers.fit_transform_prod(product_train, vocab_processor_prod, vocab)
####create train_prod mask


vocab_processor = np.zeros([len(test_data_), FLAGS.max_query_length], dtype=int)
test_data = data_helpers.fit_transform(test_data_, vocab_processor, vocab)
test_mask = np.float32(test_data > 0)

vocab_processor_prod = np.zeros([len(product_test), 4 * FLAGS.top_K, FLAGS.max_query_length_prod], dtype=int)
test_prod = data_helpers.fit_transform_prod(product_test, vocab_processor_prod, vocab)
####create test_prod mask


print("Vocabulary Size: {:d}".format(len(vocab)))

test_label_L6 = test_label[0]
train_label_L6 = train_label[0]



# Training
# ================================================== ==================================================

with tf.Graph().as_default():
    session_conf = tf.compat.v1.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.compat.v1.Session(config=session_conf)

    qp_vectors = []
    with sess.as_default():

        cnn = QPNetwork(
                sequence_length=train_data.shape[1],
                sequence_length_prod=FLAGS.max_query_length_prod,
                num_classes_L6 = 4665,
                num_quantized_chars=FLAGS.num_quantized_chars,
                word_char_length=FLAGS.max_char_length,
                vocab_size=len(vocab),
                embedding_size=FLAGS.embedding_dim,
                embedding_char_size=FLAGS.embedding_char_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                top_K = FLAGS.top_K,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                Ranking = FLAGS.Ranking,
                K = FLAGS.K,
                only_query = FLAGS.only_query

                 )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # total_parameters = 0
        # for variable in tf.compat.v1.trainable_variables():
        #     # shape is an array of tf.Dimension
        #     shape = variable.get_shape()
        #     print(shape)
        #     print(len(shape))
        #     variable_parameters = 1
        #     for dim in shape:
        #         print(dim)
        #         variable_parameters *= dim
        #     print(variable_parameters)
        #     total_parameters += variable_parameters
        # print(total_parameters)


        # Write vocabulary
        vocab_data = dict()
        vocab_data['data'] = vocab


        # Initialize all variables
        sess.run(tf.compat.v1.global_variables_initializer())

        count = 0
        vocabulary = dict()
        if FLAGS.word2vec:
            # w2v.load_model('Word2Vector.model')
            print ("\n====>len Vocab after all these {}".format(len(vocab)))
            initW = np.random.uniform(-0.25, 0.25, (len(vocab), FLAGS.embedding_dim))

            for word in vocab:
                # start = time.time()
                initW[vocab[word]] = w2v.word_vectors[word]

            # sess.run(cnn.set_W, feed_dict={cnn.place_w: initW})
            sess.run(cnn.W.assign(initW))

        def train_step(x_batch, x_mask, x_char_batch,indices_L6, values_L6, prod_batch, prod_char_batch, CM):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x_query: x_batch,
                cnn.input_x_prod: prod_batch,
                cnn.input_char_query: x_char_batch,
                cnn.input_char_prod: prod_char_batch,
                cnn.input_x_mask: x_mask,
                cnn.input_indices_L6: indices_L6,
                cnn.input_values_L6: values_L6,
                cnn.size_sample:FLAGS.batch_size,
                cnn.coocurrance_matrix: CM,
                cnn.dropout_keep_prob:FLAGS.dropout_keep_prob

            }
            _, step, loss = sess.run([train_op, global_step, cnn.loss], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))
            # train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, x_mask, x_char_batch, indices_L6, values_L6, prod_batch, prod_char_batch, CM, size_label):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x_query: x_batch,
                cnn.input_x_prod: prod_batch,
                cnn.input_char_query: x_char_batch,
                cnn.input_char_prod: prod_char_batch,
                cnn.input_x_mask: x_mask,
                cnn.input_indices_L6: indices_L6,
                cnn.input_values_L6: values_L6,
                cnn.size_sample:size_label,
                cnn.coocurrance_matrix:CM,
                cnn.dropout_keep_prob: 1.0
            }

            # step, loss, probs_L1, true_labels_L1, probs_L2, true_labels_L2, probs_L6, true_labels_L6  = sess.run([global_step, cnn.loss, cnn.probs_L1, cnn.input_y_L1, cnn.probs_L2, cnn.input_y_L2, cnn.probs_L6, cnn.input_y_L6], feed_dict)
            # print ()

            # test_f1_micro_l1 = compute_prf(probs_L6, true_labels_L6)
            # time_str = datetime.datetime.now().isoformat()
            # print("L1:  {}: step {}, loss {:g}, acc_L1 {:g}".format(time_str, step, loss, test_f1_micro_l1))

            step, loss, probs, true_labels, all_relevant_indices_1, pred_relevnat_indices_K_1, num_relevant_items_1, num_recom_items_k_1, pred_relevnat_indices_K_replace_1, all_relevant_indices_replace_1, all_relevant_indices_2, pred_relevnat_indices_K_2, num_relevant_items_2, num_recom_items_k_2, pred_relevnat_indices_K_replace_2, all_relevant_indices_replace_2, all_relevant_indices_3, pred_relevnat_indices_K_3, num_relevant_items_3, num_recom_items_k_3, pred_relevnat_indices_K_replace_3, all_relevant_indices_replace_3 = sess.run([global_step, cnn.loss, cnn.probs_L6, cnn.input_y_L6, cnn.all_relevant_indices_1, cnn.pred_relevnat_indices_K_1, cnn.num_relevant_items_1, cnn.num_recom_items_k_1, cnn.pred_relevnat_indices_K_replace_1, cnn.all_relevant_indices_replace_1, cnn.all_relevant_indices_2, cnn.pred_relevnat_indices_K_2, cnn.num_relevant_items_2,cnn.num_recom_items_k_2, cnn.pred_relevnat_indices_K_replace_2, cnn.all_relevant_indices_replace_2, cnn.all_relevant_indices_3, cnn.pred_relevnat_indices_K_3, cnn.num_relevant_items_3,cnn.num_recom_items_k_3, cnn.pred_relevnat_indices_K_replace_3, cnn.all_relevant_indices_replace_3], feed_dict)

            
#             print(pred_relevnat_indices_K_1.shape, "  ", pred_relevnat_indices_K_2.shape, "  ", pred_relevnat_indices_K_3.shape)
            
            return all_relevant_indices_1, pred_relevnat_indices_K_1, num_relevant_items_1, num_recom_items_k_1, pred_relevnat_indices_K_replace_1, all_relevant_indices_replace_1, all_relevant_indices_2, pred_relevnat_indices_K_2, num_relevant_items_2, num_recom_items_k_2, pred_relevnat_indices_K_replace_2, all_relevant_indices_replace_2, all_relevant_indices_3, pred_relevnat_indices_K_3, num_relevant_items_3, num_recom_items_k_3, pred_relevnat_indices_K_replace_3, all_relevant_indices_replace_3





        loadpath_cooccurance = "./generate_dataset/CM.pkl"
        with open(loadpath_cooccurance, 'rb') as f:
            CM = pickle.load(f)
        
        
        # Generate batches
        batches = data_helpers.batch_iter(list(zip(train_data, train_mask, train_data_char, train_label_L6, train_prod, product_train_char)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        start = time.time()
        all_acc_int = []
        for batch in batches:
            x_batch, x_mask, x_char_batch, y_batch_L6, prod_batch, prod_char_batch  = zip(*batch)

            indices_L6, values_L6 = data_helpers.generate_indice_values(y_batch_L6)

#             start = time.time()
            ################################################################################
            train_step(x_batch, x_mask, x_char_batch, indices_L6, values_L6, prod_batch, prod_char_batch, CM)
            ################################################################################
            
#             print("training step time: ", time.time() - start)

            current_step = tf.compat.v1.train.global_step(sess, global_step)


            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")

                size_dev = 2000
                batches_dev = data_helpers.dev_iter(list(zip(test_data, test_mask, test_data_char, test_label_L6, test_prod, product_test_char)), size_dev, 1, shuffle=False)

                all_relevant_indices_1_, pred_relevnat_indices_K_1_, num_relevant_items_1_, num_recom_items_k_1_, pred_relevnat_indices_K_replace_1_, all_relevant_indices_replace_1_, all_relevant_indices_2_, pred_relevnat_indices_K_2_, num_relevant_items_2_, num_recom_items_k_2_, pred_relevnat_indices_K_replace_2_, all_relevant_indices_replace_2_, all_relevant_indices_3_, pred_relevnat_indices_K_3_, num_relevant_items_3_, num_recom_items_k_3_, pred_relevnat_indices_K_replace_3_, all_relevant_indices_replace_3_= [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
                
                
                for i, batch_dev in enumerate(batches_dev):
                    print(i)
                    x_test_batch, x_test_mask, x_test_char_batch, y_test_batch_L6, prod_test_batch, prod_test_char_batch = zip(*batch_dev)


                    indices_test_L6, values_test_L6 = data_helpers.generate_indice_values(y_test_batch_L6)


                    all_relevant_indices_1, pred_relevnat_indices_K_1, relevant_items_1, num_recom_items_k_1, pred_relevnat_indices_K_replace_1, all_relevant_indices_replace_1, all_relevant_indices_2, pred_relevnat_indices_K_2, relevant_items_2, num_recom_items_k_2, pred_relevnat_indices_K_replace_2, all_relevant_indices_replace_2, all_relevant_indices_3, pred_relevnat_indices_K_3, relevant_items_3, num_recom_items_k_3, pred_relevnat_indices_K_replace_3, all_relevant_indices_replace_3 = dev_step(x_test_batch, x_test_mask, x_test_char_batch, indices_test_L6, values_test_L6, prod_test_batch, prod_test_char_batch,CM, len(x_test_batch))


                    
                    
                    all_relevant_indices_1_.extend(all_relevant_indices_1)
                    pred_relevnat_indices_K_1_.extend(pred_relevnat_indices_K_1)
                    num_relevant_items_1_.extend(relevant_items_1)
                    num_recom_items_k_1_.extend(num_recom_items_k_1)
                    pred_relevnat_indices_K_replace_1_.extend(pred_relevnat_indices_K_replace_1)
                    all_relevant_indices_replace_1_.extend(all_relevant_indices_replace_1)
                    
                    all_relevant_indices_2_.extend(all_relevant_indices_2)
                    pred_relevnat_indices_K_2_.extend(pred_relevnat_indices_K_2)
                    num_relevant_items_2_.extend(relevant_items_2)
                    num_recom_items_k_2_.extend(num_recom_items_k_2)
                    pred_relevnat_indices_K_replace_2_.extend(pred_relevnat_indices_K_replace_2)
                    all_relevant_indices_replace_2_.extend(all_relevant_indices_replace_2)
                    
                    all_relevant_indices_3_.extend(all_relevant_indices_3)
                    pred_relevnat_indices_K_3_.extend(pred_relevnat_indices_K_3)
                    num_relevant_items_3_.extend(relevant_items_3)
                    num_recom_items_k_3_.extend(num_recom_items_k_3)
                    pred_relevnat_indices_K_replace_3_.extend(pred_relevnat_indices_K_replace_3)
                    all_relevant_indices_replace_3_.extend(all_relevant_indices_replace_3)


                
                
                ################################################################################
                ################################################################################
                ################################################################################
                predicted_data = [all_relevant_indices_1_, pred_relevnat_indices_K_1_, num_relevant_items_1_, num_recom_items_k_1_, pred_relevnat_indices_K_replace_1_, all_relevant_indices_replace_1_ ]

                # with open('predictions.pickle', 'wb') as handle:
                #     pickle.dump(predicted_data, handle)

                file = open("result.txt", "a+")
                file.write('\t K == 1')

                results = fast_precision_recall_at_K(predicted_data, Bucket, minorities, k= 1)

                for key in sorted(results.keys()):
                    file.write("\nBucket  == %s " % key + "    ")
                    print("Bucket == %s " % key + "    ")
                    print("Test precision %f recall %f F1 %f" % (results[key][0], results[key][1], results[key][2]))
                    file.write("\n===Test precision %f recall %f F1 %f" % (results[key][0], results[key][1], results[key][2]))
                    file.write('\n')


                P,R, F = 0.0,0.0,0.0
                for key in sorted(results.keys()):
                    P += results[key][0]
                    R += results[key][1]
                    F += results[key][2]

                num_buckets = len(results.keys())
                print("\nTest Macro_precision %f Macro_recall %f Macro_F1 %f" % (P/num_buckets, R/num_buckets, F/num_buckets))
                file.write("\nTest Macro_precision %f Macro_recall %f Macro_F1 %f" % (P/num_buckets, R/num_buckets, F/num_buckets))
                file.write('\n')
                
                P_micro,R_micro, F_micro = 0.0,0.0,0.0
                for key in sorted(results.keys()):
                    P_micro += (results[key][0] * results[key][3])
                    R_micro += (results[key][1] * results[key][3])
                
                
                F1_micro = (2 * P_micro * R_micro) / (P_micro + R_micro)
                file.write("Test Micro_precision %f Micro_recall %f Micro_F1 %f" % ( P_micro, R_micro, F1_micro))
                file.write('\n\n')

                ################################################################################
                ################################################################################
                ################################################################################
               
                predicted_data = [all_relevant_indices_2_, pred_relevnat_indices_K_2_, num_relevant_items_2_, num_recom_items_k_2_, pred_relevnat_indices_K_replace_2_, all_relevant_indices_replace_2_ ]


#                 file = open("result.txt", "a+")
                file.write('\t K == 2')

                results = fast_precision_recall_at_K(predicted_data, Bucket, minorities, k= 2)

                for key in sorted(results.keys()):
                    file.write("\nBucket  == %s " % key + "    ")
                    print("Bucket == %s " % key + "    ")
                    print("Test precision %f recall %f F1 %f" % (results[key][0], results[key][1], results[key][2]))
                    file.write("\n===Test precision %f recall %f F1 %f" % (results[key][0], results[key][1], results[key][2]))
                    file.write('\n')


                P,R, F = 0.0,0.0,0.0
                for key in sorted(results.keys()):
                    P += results[key][0]
                    R += results[key][1]
                    F += results[key][2]

                num_buckets = len(results.keys())
                print("\nTest Macro_precision %f Macro_recall %f Macro_F1 %f" % (P/num_buckets, R/num_buckets, F/num_buckets))
                file.write("\nTest Macro_precision %f Macro_recall %f Macro_F1 %f" % (P/num_buckets, R/num_buckets, F/num_buckets))
                file.write('\n')
                
                P_micro,R_micro, F_micro = 0.0,0.0,0.0
                for key in sorted(results.keys()):
                    P_micro += (results[key][0] * results[key][3])
                    R_micro += (results[key][1] * results[key][3])
                
                
                F1_micro = (2 * P_micro * R_micro) / (P_micro + R_micro)
                file.write("Test Micro_precision %f Micro_recall %f Micro_F1 %f" % ( P_micro, R_micro, F1_micro))
                file.write('\n\n')
                
                
                
                ################################################################################
                ################################################################################
                ################################################################################
                
                predicted_data = [all_relevant_indices_3_, pred_relevnat_indices_K_3_, num_relevant_items_3_, num_recom_items_k_3_, pred_relevnat_indices_K_replace_3_, all_relevant_indices_replace_3_ ]


#                 file = open("result.txt", "a+")
                file.write('\t K == 3')
                results = fast_precision_recall_at_K(predicted_data, Bucket, minorities, k= 3)

                for key in sorted(results.keys()):
                    file.write("\nBucket  == %s " % key + "    ")
                    print("Bucket == %s " % key + "    ")
                    print("Test precision %f recall %f F1 %f" % (results[key][0], results[key][1], results[key][2]))
                    file.write("\n===Test precision %f recall %f F1 %f" % (results[key][0], results[key][1], results[key][2]))
                    file.write('\n')


                P,R, F = 0.0,0.0,0.0
                for key in sorted(results.keys()):
                    P += results[key][0]
                    R += results[key][1]
                    F += results[key][2]

                num_buckets = len(results.keys())
                print("\nTest Macro_precision %f Macro_recall %f Macro_F1 %f" % (P/num_buckets, R/num_buckets, F/num_buckets))
                file.write("\nTest Macro_precision %f Macro_recall %f Macro_F1 %f" % (P/num_buckets, R/num_buckets, F/num_buckets))
                file.write('\n')
                
                P_micro,R_micro, F_micro = 0.0,0.0,0.0
                for key in sorted(results.keys()):
                    P_micro += (results[key][0] * results[key][3])
                    R_micro += (results[key][1] * results[key][3])
                
                F1_micro = (2 * P_micro * R_micro) / (P_micro + R_micro)
                file.write("Test Micro_precision %f Micro_recall %f Micro_F1 %f" % ( P_micro, R_micro, F1_micro))
                file.write('\n\n\n\n')
            
                
                print('')
                file.close()


