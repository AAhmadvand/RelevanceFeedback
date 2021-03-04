import numpy as np
import re, json, csv
from sklearn.utils import shuffle
from collections import defaultdict, Counter
from embedding import Word2Vec



################################################################################################
def char2vec(text, sequence_max_length, char_dict):
    data = np.zeros(sequence_max_length)
    for i in range(0, len(text)):
        if i > sequence_max_length:
            return data
        elif text[i] in char_dict:
            try:
                data[i] = char_dict[text[i]]
            except:
                pass
        else:
            # unknown character set to be 68
            try:
                data[i] = 41
            except:
                pass
    return data
################################################################################################
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789'><().-/:;#,_ "
char_dict = {}
for i, c in enumerate(alphabet):
    char_dict[c] = i + 1

################################################################################################
def load_data_and_labels(max_query_length_prod, max_query_length, max_char_length, top_K, EXPANSION, only_query, minoriy_class_result):


    vocabulary = []
    vocabulary_doc = []

    train_data1 = open("./generate_dataset/train_dataset_bm25_.txt", 'r').read().split('\n')
    print (len(train_data1))
    test_data1 = open("./generate_dataset/validation_set_bucket_bm25_.txt", 'r').read().split('\n')
    print (len(test_data1))
    
    
    ####################Expanded Queries##################
    with open('./generate_dataset/map_query_expanded_validation.json', 'r') as handel:
        expanded_query_test = json.load(handel)

    handel.close()
    
    print(len(expanded_query_test.keys()))

    with open('./generate_dataset/map_query_expanded_train.json','r') as handel:
        expanded_query_train = json.load(handel)

    handel.close()

    print("data is read....")

    train_data_char = []
    train_data = []
    product_train_char = []
    product_train = []
    train_check_numeric = []
    train_label_L1 = []
    train_label_L2 = []
    train_label_L3 = []
    train_label_L4 = []
    train_label_L5 = []
    train_label_L6 = []
    all_relevant_categories_l1 = 0
    all_relevant_categories_l6 = 0
    fequencies_classes = Counter()

    D = []
    counter = 0
    max_len_query = 0
    sum_q_len_train = 0
    for i, sample in enumerate(train_data1):
        if i % 100000 == 0:
            print (i)
#         if i > 10000:
#             break

        if len(sample) > 0:
            line = json.loads(sample)
            labels = line["label"]

            ex_query = line["query"].lower()
            
            ##EXPANSION EXPANSION EXPANSION EXPANSION
#             ex_query = expanded_query_train[str(i)]  
#             if i == 0:
#                 print(line["query"].lower(),"    ", ex_query)
            
            # print (ex_query, "-====")
            new_sample = "<bos> " + ex_query + " <eos>"
            l_q = len(line["query"].split())
            if l_q > max_len_query:
                max_len_query = l_q

            sum_q_len_train += l_q
            train_check_numeric.append(checkout_dims(new_sample))
            train_data.append(new_sample)

            D.append(len(new_sample.split()))

            prod = []
            try:
                short_descs = line["short_disc"]
#                 print(short_descs)
            except:
                short_descs = [new_sample]

            
#             if not only_query:
            for ind_desc in range(len(short_descs)):
                for field in short_descs[ind_desc]:
                    vocabulary_doc.extend(field.split())
                                
                                
                                
            for ind_desc in range(top_K):
                try:
                    for field in short_descs[ind_desc]:
                        prod.append(field)
                except:
                    for i in range(4):
                          prod.append(field)


            product_train.append(prod)
            train_label_L6.append(line["label"])
            for ind, val in enumerate(line["label"]["values"]):
                 if val > 10e-7:
#                     print(line["label"]["indexes"][ind])
                    fequencies_classes[line["label"]["indexes"][ind]] += 1

#             all_relevant_categories_l1 += np.sum(np.asarray(line["taxonomy"]["L1"][1])>1e-5)
            all_relevant_categories_l6 += np.sum(np.asarray(line["label"]["values"])>1e-7)


            #####Compute query char representations
            vocabulary.extend(new_sample.split())
            word_char = np.zeros([max_query_length, max_char_length], dtype=int)
            for j, word in enumerate(new_sample.split()):
                try:
                    word_char[j] = char2vec(word, max_char_length, char_dict)
                except:
                    pass

            train_data_char.append(word_char)
            #####Compute Product description char representations
            prod_char = []
            for desc in prod:
                
                word_char = np.zeros([max_query_length_prod, max_char_length], dtype=int)
                desc = desc.strip()
                for j, word in enumerate(desc.split()):
                    try:
                        word_char[j] = char2vec(word, max_char_length, char_dict)
                    except:
                        pass
                prod_char.append(word_char)

#             print(prod_char)
            product_train_char.append(np.array(prod_char))

    print("all_relevant_categories_l1: ", all_relevant_categories_l1/ len(product_train_char))
    print("all_relevant_categories_l6: ", all_relevant_categories_l6/ len(product_train_char))
    print("len_vocab", len(vocabulary))
    
    sorted_clss_freq = fequencies_classes.most_common()
    ##the last labels
    print("#####################Compute_minority_classes#####################" )
    if minoriy_class_result:
        print("sorted_clss_freq: ", len(sorted_clss_freq))
#         print(sorted_clss_freq)
        sum_frequencies = sum(np.array([item[1] for item in sorted_clss_freq]))
        avg = sum_frequencies/len(fequencies_classes)
        minorities = np.array([item[0] for item in sorted_clss_freq if item[1] < avg])
        print("sum_frequencies: ", sum_frequencies, "   ", "avg_click: ", avg, "len minority classes: ", len(minorities) )
    else:
        ###all classes here are used
        minorities = np.array([item[0] for item in sorted_clss_freq])
    print("#####################")

    
    
    print("avg len of query is : ", np.mean(np.asarray(D)))
    train_label = [train_label_L6]


    test_data_char = []
    test_data = []
    product_test_char = []
    product_test = []
    test_check_numeric = []
    test_label_L1 = []
    test_label_L2 = []
    test_label_L3 = []
    test_label_L4 = []
    test_label_L5 = []
    test_label_L6 = []
    Bucket = defaultdict(list)

    print("max_len_query train:", max_len_query)
    print("Avg_len_query train:", sum_q_len_train/len(product_train_char))
    print("len_train_set:", len(train_label_L6))


    print ("num samples with no labels:", counter)
    all_categories = []
    max_len_query = 0
    sum_q_len_test = 0
    vocab_test = set()

    count_samp = 0
    for i, sample in enumerate(test_data1):
        
#         if i > 1000:
#             break

        if len(sample) > 0:

                line = json.loads(sample)
                labels = line["label"]

                ex_query = line["query"].lower()
                
                ##EXPANSION EXPANSION EXPANSION EXPANSION
#                 ex_query = expanded_query_test[str(i)]
#                 if i == 0:
#                     print(line["query"].lower(), "  ", ex_query)
                
                
                # print (query , ex_query)
                new_sample = "<bos> " + ex_query + " <eos>"

                l_q = len(line["query"].split())
                if l_q > max_len_query:
                    max_len_query = l_q

                sum_q_len_test += l_q
                test_check_numeric.append(checkout_dims(new_sample))
                test_data.append(new_sample)

                vocab_test.update(new_sample.split())

                prod = []
                try:
                    short_descs = line["short_disc"]
                except:
                    short_descs = [new_sample]
                    
                    

#                 if not only_query:
                for ind_desc in range(len(short_descs)):
                    for field in short_descs[ind_desc]:
                        vocabulary_doc.extend(field.split())
                   
                for ind_desc in range(top_K):
                    try:
                        for field in short_descs[ind_desc]:
                             prod.append(field)
                    except:
                        for i in range(4):
                            prod.append(field)
                
#                 print(len(prod))
                product_test.append(prod)
                test_label_L6.append(line["label"])
                

                #####Compute query char representations
                word_char = np.zeros([max_query_length, max_char_length], dtype=int)
                for j, word in enumerate(new_sample.split()):
                    try:
                        word_char[j] = char2vec(word, max_char_length, char_dict)
                    except:
                        pass

                test_data_char.append(word_char)

                #####Compute Product description char representations
                prod_char = []
                for desc in prod:
                    word_char = np.zeros([max_query_length_prod, max_char_length], dtype=int)
#                     print (desc)
                    desc = desc.strip()
                    for j, word in enumerate(desc.split()):
                        try:
                            word_char[j] = char2vec(word, max_char_length, char_dict)
                        except:
                            pass
                    prod_char.append(word_char)

#                 print(len(prod_char))
                product_test_char.append(np.array(prod_char))


                
#                 print (line["Bucket"])   
                for buck in line["Bucket"]:
                     Bucket[buck].append(count_samp)
                count_samp += 1
#                 print (Bucket)

    test_label = [test_label_L6]
    print("len(vocab_test): ", len(vocab_test))
    print("max_len_query test:", max_len_query)
    print("Avg_len_query test:", sum_q_len_test/len(product_test_char))
    print("len test set: ", len(test_label_L6))

    counts_tokens = Counter(vocabulary)
    vocab = {}
    words_ = set()
    set_words = set(vocabulary)
    count = 0
    for item in sorted(set_words):
        if counts_tokens[item] > 5:
            vocab[item] = count
            words_.add(item)
            count += 1
            
    counts_tokens = Counter(vocabulary_doc)
    set_words = set(vocabulary_doc)
    for item in sorted(set_words):
        if counts_tokens[item] > 25 and item not in words_:
            vocab[item] = count
            count += 1

    
    w2v = Word2Vec('/home/jupyter/GoogleNews-vectors-negative300.bin', vocab.keys())
#     w2v = {}
    
    # Read CSV Info
    return train_data, np.array(train_data_char), train_label, test_data, np.array(test_data_char), test_label ,\
           vocab,  product_train, product_train_char, product_test, product_test_char, Bucket, w2v, minorities


################################################################################################
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_ = np.array(data)
    data_size_ = len(data_)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    data_size = num_batches_per_epoch * batch_size

    for ind in range(data_size - data_size_):
        data.append(data[-1])

    data = np.array(data)

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


################################################################################################
def fit_transform(train_data, vocab_processor, vocab):

    for iter, line in enumerate(train_data):
        text = line.strip()
        for j, word in enumerate(text.split()):
            try:
                vocab_processor[iter][j] = vocab[word]
            except:
                pass


    return vocab_processor
################################################################################################
def fit_transform_prod(train_data, vocab_processor, vocab):
    for iter, line in enumerate(train_data):
        for i, sample in enumerate(line):
            text = sample.strip()
            for j, word in enumerate(text.split()):
                try:
                    vocab_processor[iter][i][j] = vocab[word]
                except:
                    pass


    return vocab_processor

################################################################################################
def checkout_dims (query):
    rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", query)
    if len(rr) > 0:
        return [1,0]
    else:
        return [0,1]

def generate_indice_values (label):
    indices = []
    values = []
    for ind, indice in enumerate(label):
        for sample in indice['indexes']:
            indices.append([ind, sample])

    for val in label:
        values.extend(val['values'])


    return indices, values


################################################################################################
def dev_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]
