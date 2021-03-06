import re, json
from pyserini import index
#
#
#
#
index_reader = index.IndexReader('/Users/axa8yjv/Documents/Hierarchical_classifier/Relevance_Feedback/RF/dataset/indexes/')
dataset=open('/Users/axa8yjv/Documents/Hierarchical_classifier/Relevance_Feedback/RF/dataset/log/result_train.log', 'r').read().split('\n')
map_querytoexpanded = {}
q = []
expanded_q = []
count = 0
test_data1 = open("/Users/axa8yjv/Downloads/train_dataset_10.txt", 'r').read().split('\n')
test = []
for i, sample in enumerate(test_data1):


    if len(sample) > 0:

        line = json.loads(sample)
        query = line["query"]
        test.append(query)


count_test = 0
for line in dataset:
    if 'Original Query:' in line:
        query = test[count]
        count_test +=1

    elif 'Running new query:' in line:
        expand = line.split(':')[-1]
        expand = expand.split()
        new_tokens = {}
        for ex in expand:
            try:
                word = re.findall('\(.*?\)',ex)[0]
                val = float(ex.split('^')[1])
                clean_word = word[word.find("(")+1:word.find(")")]
                new_tokens[clean_word] = val
            except:
                print(line)

        new_tokens = sorted(new_tokens.items(), key=lambda x: x[1], reverse=True)
        expnded_query = []
        query_token = query.split()

        expnded_query.extend(query_token)
        expnded_query.extend(["</extra>"])
        analyzed = index_reader.analyze(query)

        for t in new_tokens:
            if t[0] not in analyzed:
                expnded_query.append(t[0])

        expanded_q.append(expnded_query)

        map_querytoexpanded[count] = ' '.join(expnded_query)
        count+=1


with open('/Users/axa8yjv/Documents/Hierarchical_classifier/Relevance_Feedback/RF/dataset/expanded_data/map_query_expanded_train.json', 'w') as handel:
    json.dump(map_querytoexpanded, handel)

handel.close()

print("Training set is generted...")

##################################################################
##################################################################
##################################################################



dataset=open('/Users/axa8yjv/Documents/Hierarchical_classifier/Relevance_Feedback/RF/dataset/log/result_validation.log', 'r')
map_querytoexpanded_train = {}
q = []
expanded_q = []
count = 0
train_data1 = open("/Users/axa8yjv/Downloads/validation_set_bucket_10.txt", 'r').read().split('\n')
train = []
for i, sample in enumerate(train_data1):


    if len(sample) > 0:
        line = json.loads(sample)
        query = line["query"]
        train.append(query)


count_train = 0
for line in dataset:
    if 'Original Query:' in line:
        query = train[count]
        count_train +=1

    elif 'Running new query:' in line:
        expand = line.split('Running new query:')[-1]
        expand = expand.split()
        new_tokens = {}
        for ex in expand:
            word = re.findall('\(.*?\)', ex)[0]

            val = float(ex.split('^')[1])
            clean_word = word[word.find("(") + 1:word.find(")")]
            new_tokens[clean_word] = val
            h = 0

        new_tokens = sorted(new_tokens.items(), key=lambda x: x[1], reverse=True)
        expnded_query = []
        query_token = query.split()

        expnded_query.extend(query_token)
        expnded_query.extend(["</extra>"])
        analyzed = index_reader.analyze(query)
        for t in new_tokens:
            if t[0] not in analyzed:
                expnded_query.append(t[0])

        expanded_q.append(expnded_query)

        map_querytoexpanded_train[count] = ' '.join(expnded_query)
        count+=1


with open('/Users/axa8yjv/Documents/Hierarchical_classifier/Relevance_Feedback/RF/dataset/expanded_data/map_query_expanded_validation.json', 'w') as handel:
    json.dump(map_querytoexpanded_train, handel)
handel.close()
print("Trest set is generted...")




##################################################################
##################################################################
##################################################################



dataset=open('/Users/axa8yjv/Documents/Hierarchical_classifier/Relevance_Feedback/RF/dataset/log/result_validation_segments.log', 'r')
map_querytoexpanded_train = {}
q = []
expanded_q = []
count = 0
train_data1 = open("/Users/axa8yjv/Downloads/validation_set_bucket_segments.txt", 'r').read().split('\n')
train = []
for i, sample in enumerate(train_data1):


    if len(sample) > 0:
        line = json.loads(sample)
        query = line["query"]
        train.append(query)


count_train = 0
for line in dataset:
    if 'Original Query:' in line:
        query = train[count]
        count_train +=1

    elif 'Running new query:' in line:
        expand = line.split('Running new query:')[-1]
        expand = expand.split()
        new_tokens = {}
        for ex in expand:
            word = re.findall('\(.*?\)', ex)[0]

            val = float(ex.split('^')[1])
            clean_word = word[word.find("(") + 1:word.find(")")]
            new_tokens[clean_word] = val
            h = 0

        new_tokens = sorted(new_tokens.items(), key=lambda x: x[1], reverse=True)
        expnded_query = []
        query_token = query.split()

        expnded_query.extend(query_token)
        expnded_query.extend(["</extra>"])
        analyzed = index_reader.analyze(query)
        for t in new_tokens:
            if t[0] not in analyzed:
                expnded_query.append(t[0])

        expanded_q.append(expnded_query)

        map_querytoexpanded_train[count] = ' '.join(expnded_query)
        count+=1


with open('/Users/axa8yjv/Documents/Hierarchical_classifier/Relevance_Feedback/RF/dataset/expanded_data/map_query_expanded_validation_segments.json', 'w') as handel:
    json.dump(map_querytoexpanded_train, handel)
handel.close()
print("Trest_segments set is generted...")