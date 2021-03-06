import logging,json, itertools
from pyserini.search import SimpleSearcher


####Generation of the indexes
###Second step == > python3 -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator
# -threads 1 -input dataset/collection/ -index dataset/indexes/sample_collection_jsonl -storePositions -storeDocvectors -storeRaw



###Third step needs to be run in command line  ====> python3 Generate_result.py > dataset/log/result_train.log




# searcher = SimpleSearcher('/Users/axa8yjv/Documents/Hierarchical_classifier/Relevance_Feedback/RF/dataset/indexes/')
# test_data1 = open("/Users/axa8yjv/Downloads/train_dataset_10.txt", 'r').read().split('\n')
# for i, sample in enumerate(test_data1):
#
#     if len(sample) > 0:
#         line = json.loads(sample)
#         query = line["query"]
#         searcher.set_bm25(k1=1.5, b=0.75)
#         searcher.set_rm3(15, 10, original_query_weight=float(0.5), rm3_output_query=True)
#         hits = searcher.search(query)







searcher = SimpleSearcher('/Users/axa8yjv/Documents/Hierarchical_classifier/Relevance_Feedback/RF/dataset/indexes/')
train_data1 = open("/Users/axa8yjv/Downloads/validation_set_bucket_10.txt", 'r').read().split('\n')
print(len(train_data1))
for i, sample in enumerate(train_data1):
    if len(sample) > 0:
        line = json.loads(sample)
        query = line["query"]
        searcher.set_bm25(k1=1.5, b=0.75)
        searcher.set_rm3(15, 10, original_query_weight=float(0.5), rm3_output_query=True)
        hits = searcher.search(query)