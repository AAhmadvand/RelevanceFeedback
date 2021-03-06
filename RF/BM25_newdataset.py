import logging, json, itertools
from pyserini.search import SimpleSearcher
import time

searcher = SimpleSearcher('/Users/axa8yjv/Documents/Hierarchical_classifier/Relevance_Feedback/RF/dataset/indexes/')


train_data = open("/Users/axa8yjv/Downloads/train_dataset_10.txt", 'r').read().split('\n')
train_data_bm25 = open("/Users/axa8yjv/Downloads/train_dataset_bm25.txt", 'w')

for i, sample in enumerate(train_data):

    if len(sample) > 0:
        new_s = {}

        line = json.loads(sample)
        # new_s = line
        query = line["query"]
        searcher.set_bm25(k1=1.5, b=0.75)
        hits = searcher.search(query)
        new_discs = []
        values = []
        # print(len(hits))
        for i in range(len(hits)):
             values = searcher.doc(hits[i].docid).raw().split(" . ")[0:4]
             new_discs.append(values)

        new_s["query"] = query
        new_s["label"] = line["label"]
        # print(query)
        # print(new_discs)
        new_s["short_disc"] = new_discs
        # print(line["short_disc"])
        train_data_bm25.write(json.dumps(new_s))
        train_data_bm25.write('\n')


train_data_bm25.close()


print("Train is done ....")


test_data = open("/Users/axa8yjv/Downloads/validation_set_bucket_10.txt", 'r').read().split('\n')
test_data_bm25 = open("/Users/axa8yjv/Downloads/validation_set_bucket_bm25.txt", 'w')

for i, sample in enumerate(test_data):

    if len(sample) > 0:
        new_s = {}

        line = json.loads(sample)
        # new_s = line
        query = line["query"]
        searcher.set_bm25(k1=1.5, b=0.75)
        hits = searcher.search(query)
        new_discs = []
        for i in range(len(hits)):
            values = searcher.doc(hits[i].docid).raw().split(" . ")[0:4]
            new_discs.append(values)

        new_s["query"] = query
        new_s["label"] = line["label"]
        new_s["Bucket"] = line["Bucket"]
        # print(query)
        # print(new_discs)
        new_s["short_disc"] = new_discs
        # print(line["short_disc"])
        test_data_bm25.write(json.dumps(new_s))
        test_data_bm25.write('\n')


test_data_bm25.close()
#

print("validation is done ....")

test_data_seg = open("/Users/axa8yjv/Downloads/validation_set_bucket_segments.txt", 'r').read().split('\n')
test_data_bm25_seg = open("/Users/axa8yjv/Downloads/validation_dataset_segments_bm25.txt", 'w')

start = time.time()
for i, sample in enumerate(test_data_seg):

    if len(sample) > 0:
        new_s = {}

        line = json.loads(sample)
        # new_s = line
        query = line["query"]
        searcher.set_bm25(k1=1.5, b=0.75)
        hits = searcher.search(query)
        new_discs = []
        for i in range(len(hits)):
            values = searcher.doc(hits[i].docid).raw().split(" . ")[0:4]
            new_discs.append(values)

        new_s["query"] = query
        new_s["label"] = line["label"]
        new_s["Bucket"] = line["Bucket"]
        # print(query)
        # print(new_discs)
        new_s["short_disc"] = new_discs
        # print(line["short_disc"])
        test_data_bm25_seg.write(json.dumps(new_s))
        test_data_bm25_seg.write('\n')


test_data_bm25_seg.close()

print(time.time() - start)