import time, json
import logging
from pyserini.search import SimpleSearcher

########First step need to be run


train_data1 = open("/Users/axa8yjv/Downloads/train_dataset_10.txt", 'r').read().split('\n')

validation = open("/Users/axa8yjv/Downloads/validation_set_bucket_10.txt", 'r').read().split('\n')
train_data1.extend(validation)

# validation = open("/Users/axa8yjv/Downloads/validation_set_bucket_segments.txt", 'r').read().split('\n')
# train_data1.extend(validation)

print(len(train_data1))

all_docs = set()

docid = 0
file_name = open("/Users/axa8yjv/Documents/Hierarchical_classifier/Relevance_Feedback/RF/dataset/collection/documents.jsonl", 'w')
for i, sample in enumerate(train_data1):
    if i % 100000 == 0:
        print(i)

    if len(sample) > 0:
        line = json.loads(sample)
        data = {}
        query = line["query"]
        docs = line["short_disc"]
        # print (docs)
        for doc in docs:
            contents = " . ".join(doc)
            data['id'] = docid
            if contents not in all_docs:
                all_docs.add(contents)

                data['contents'] = str(contents)
                # print (contents)
                # try:
                file_name.write(json.dumps(data))
                # except:
                #     file_name.write(json.dumps({'id':docid , 'contents':contents.split('.')[0]}))
                file_name.write('\n')
                docid += 1
            else:
                pass
print(docid)
file_name.close()

