import numpy as np
from collections import defaultdict


def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)
def macro_precision(yhat, y, minorities):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    num = num[minorities]
    print (num)
    return num, np.mean(num)

def macro_recall(yhat, y, minorities):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    num = num[minorities]
    print (num)
    return num, np.mean(num)

def macro_f1(yhat, y, minorities):
    num_p, prec = macro_precision(yhat, y, minorities)
    num_r, rec = macro_recall(yhat, y, minorities)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = (2*(prec*rec))/(prec+rec)

    f1s = []
    for i, p in enumerate(num_p):
        if p + num_r[i] == 0:
            f1s.append(0.)
        else:
            f1s.append((2 * (p * num_r[i])) / (p + num_r[i]))


    return [prec, rec, f1, f1s]


def precision_at_k(yhat_raw, y, k):

    std = np.std(yhat_raw, axis = 1)
    mean = np.mean(yhat_raw, axis = 1)
    threshold = mean + 2 * std

    #num true labels in top k predictions / k
    sortd_pre = np.argsort(yhat_raw)[:,::-1]
    topk_pre = sortd_pre[:,:k]

    topk_pre_20 = sortd_pre[:, :20]


    #get precision at k for each example
    vals_precision = []
    vals_recall = []
    vals_f1 = []
    for i, top_k in enumerate(topk_pre):

        ### I added all numbers with 1 to avoind zero multiplication

        # =======================================
        all_relevants = sum((y[i] > 1e-7))
        if all_relevants == 0:
            continue

        relevant_items = (y[i] > 1e-7) * 1
        all_relevant_classes = np.asarray(np.nonzero(relevant_items))[0] + 1  ### I added all numbers with 1 to avoind zero multiplication

        # =======================================
        # =======================================
        pred_relevant_items = (yhat_raw[i][top_k] > 1/(2*k)) * 1
        pred_relevant_items[pred_relevant_items == 0] = -1000
        num_recom_items_k = sum(pred_relevant_items>0)
        predicted_items_at_k = pred_relevant_items * (top_k + 1)  ### I added all numbers with 1 to avoind zero multiplication
        # =======================================
        # =======================================
        hit_set_at_k = set(all_relevant_classes).intersection(predicted_items_at_k)
        # =======================================
        # =======================================

        #Compute precision@K
        if len(hit_set_at_k) >= k:
            precision_k = 1
        elif num_recom_items_k == 0:
            precision_k = 0
        else:
            precision_k = len(hit_set_at_k) / num_recom_items_k
        # =======================================
        # =======================================


        #Compute recall@K
        recall_k = len(hit_set_at_k) / (all_relevants + 1e-12)
        # =======================================
        # =======================================
        vals_precision.append(precision_k)
        vals_recall.append( recall_k)

        # print (num_recom_items_k, all_relevants)


    return np.mean(vals_precision), np.mean(vals_recall)

# def precision_at_k(yhat_raw, y, k):
#
#     std = np.std(yhat_raw, axis = 1)
#     mean = np.mean(yhat_raw, axis = 1)
#     threshold = mean + 2 * std
#
#     #num true labels in top k predictions / k
#     sortd_pre = np.argsort(yhat_raw)[:,::-1]
#     topk_pre = sortd_pre[:,:k]
#
#     topk_pre_20 = sortd_pre[:, :20]
#
#
#     #get precision at k for each example
#     vals_precision = []
#     vals_recall = []
#     vals_f1 = []
#     for i, top_k in enumerate(topk_pre):
#
#         ### I added all numbers with 1 to avoind zero multiplication
#         ### I added all numbers with 1 to avoind zero multiplication
#         ### I added all numbers with 1 to avoind zero multiplication
#
#         # =======================================
#         all_relevants = sum((y[i] > 1e-5))
#         if all_relevants == 0:
#             continue
#         # ======================================
#
#         relevant_items = (y[i] > 1e-5) * 1
#         all_relevant_classes = np.asarray(np.nonzero(relevant_items))[0] + 1  ### I added all numbers with 1 to avoind zero multiplication
#
#         # =======================================
#         relevant_items = (y[i][top_k] > 1e-5) * 1
#         relevant_items[relevant_items == 0] = -1
#         true_classes_at_k = relevant_items * (top_k +1) ### I added all numbers with 1 to avoind zero multiplication
#         # =======================================
#
#
#         # =======================================
#         # pred_relevant_items = (yhat_raw[i][top_k] > 1/(k*2)) * 1
#         # pred_relevant_items[pred_relevant_items == 0] = -1000
#         # predicted_items_at_k = pred_relevant_items * (top_k +1) ### I added all numbers with 1 to avoind zero multiplication
#         # =======================================
#
#         # =======================================
#         predicted_items_at_k = (topk_pre[i]+1) ### I added all numbers with 1 to avoind zero multiplication
#         # predicted_items_at_k = (top_k+1) ### I added all numbers with 1 to avoind zero multiplication
#
#         # =======================================
#
#         # num_relevant_items_at_k = sum((y[i][topk_tru[i]] > 1e-5) * 1)
#
#         hit_set = set(all_relevant_classes).intersection(predicted_items_at_k)
#         if len(hit_set) >= k:
#            precision = 1
#         else:
#             precision = len(hit_set) / float(k)
#
#         vals_precision.append(precision)
#
#
#         hit_set_at_k = set(true_classes_at_k).intersection(predicted_items_at_k)
#         vals_recall.append(  len(hit_set_at_k) / (all_relevants+1e-10))
#
#     return np.mean(vals_precision), np.mean(vals_recall)

#
# def precision_at_k(yhat_raw, y, k):
#
#
#     #num true labels in top k predictions / k
#     sortd_pre = np.argsort(yhat_raw)[:,::-1]
#     topk_pre = sortd_pre[:,:k]
#
#
#     predicted = []
#     true = []
#
#     for i, top_k in enumerate(topk_pre):
#         true.append((y[i][top_k] > 1e-5) * 1)
#         predicted.append(np.ones(len(top_k)))
#
#     return micro_f1(np.asarray(predicted).ravel() > 0.5, np.asarray(true).ravel())
#

def fast_precision_recall_at_K_minorities(pred_true, Bucket, minorities, k):

    all_relevant_indices_, pred_relevnat_indices_K_, num_relevant_items_, num_recom_items_k_, pred_relevnat_indices_K_replace_, all_relevant_indices_replace_ = \
        np.array(pred_true[0]), np.array(pred_true[1]),np.array(pred_true[2]),np.array(pred_true[3]),np.array(pred_true[4]),np.array(pred_true[5])


    results = {}
    vals = {}

    for key in sorted(Bucket.keys()):
        
        dist = 0
        vals_precision = []
        vals_recall = []
        vals_f1 = []
        
        sample_all_relevant_indices_ = all_relevant_indices_[np.array(Bucket[key])]
#         print(len(sample_all_relevant_indices_))
        
        sample_pred_relevnat_indices_K_replace_ = pred_relevnat_indices_K_replace_[np.array(Bucket[key])]
#         print(sample_pred_relevnat_indices_K_replace_)

        sample_all_relevant_indices_replace_ = all_relevant_indices_replace_[np.array(Bucket[key])]
        sample_num_recom_items_k_ = num_recom_items_k_[np.array(Bucket[key])]
        sample_num_relevant_items_ = num_relevant_items_[np.array(Bucket[key])]

        for i, sample in enumerate(sample_all_relevant_indices_):
            predicted_items_at_k = sample_pred_relevnat_indices_K_replace_[i]
            all_relevant_classes = sample_all_relevant_indices_replace_[i]
            num_recom_items_k = sample_num_recom_items_k_[i]
            all_relevants = sample_num_relevant_items_[i]

            
            
            minority_relevant_classes = set(all_relevant_classes).intersection(minorities)
            
            if (len(minority_relevant_classes) > 0):
                dist += 1
            
    #             hit_set_at_k = set(all_relevant_classes).intersection(predicted_items_at_k)
                hit_set_at_k = set(minority_relevant_classes).intersection(predicted_items_at_k)
                # Compute precision@K
            
                if len(hit_set_at_k) >= k:
                    precision_k = 1
    #             elif num_recom_items_k == 0:
    #                 precision_k = 0
                else:
                    precision_k = len(hit_set_at_k) / k

                # Compute recall@K
    #             recall_k = len(hit_set_at_k) / (all_relevants + 1e-12)
                recall_k = len(hit_set_at_k) / (len(minority_relevant_classes) + 1e-12)
                # =======================================
                # =======================================
                vals_precision.append(precision_k)
                vals_recall.append(recall_k)
                if precision_k + recall_k > 0:
                    vals_f1.append((2 * precision_k * recall_k) / (precision_k + recall_k))
                else:
                    vals_f1.append(0)
        
        vals[key] = dist
        results[key] = [np.mean(vals_precision), np.mean(vals_recall), np.mean(vals_f1), 0 ]

    for key in results:
#         print (vals)
        print(vals[key] / sum(vals.values()))
        results[key][3] = vals[key] / sum(vals.values())
             
    return results


def fast_precision_recall_at_K(pred_true, Bucket, minorities, k):

    all_relevant_indices_, pred_relevnat_indices_K_, num_relevant_items_, num_recom_items_k_, pred_relevnat_indices_K_replace_, all_relevant_indices_replace_ = \
        np.array(pred_true[0]), np.array(pred_true[1]),np.array(pred_true[2]),np.array(pred_true[3]),np.array(pred_true[4]),np.array(pred_true[5])


    results = {}

    for key in sorted(Bucket.keys()):
        
        
        vals_precision = []
        vals_recall = []
        vals_f1 = []
        
        sample_all_relevant_indices_ = all_relevant_indices_[np.array(Bucket[key])]
        sample_pred_relevnat_indices_K_replace_ = pred_relevnat_indices_K_replace_[np.array(Bucket[key])]
#         print(sample_pred_relevnat_indices_K_replace_)

        sample_all_relevant_indices_replace_ = all_relevant_indices_replace_[np.array(Bucket[key])]
        sample_num_recom_items_k_ = num_recom_items_k_[np.array(Bucket[key])]
        sample_num_relevant_items_ = num_relevant_items_[np.array(Bucket[key])]

        for i, sample in enumerate(sample_all_relevant_indices_):
            predicted_items_at_k = sample_pred_relevnat_indices_K_replace_[i]
            all_relevant_classes = sample_all_relevant_indices_replace_[i]
            num_recom_items_k = sample_num_recom_items_k_[i]
            all_relevants = sample_num_relevant_items_[i]

            hit_set_at_k = set(all_relevant_classes).intersection(predicted_items_at_k)

            # Compute precision@K
            if len(hit_set_at_k) >= k:
                precision_k = 1
#             elif num_recom_items_k == 0:
#                 precision_k = 0
            else:
                precision_k = len(hit_set_at_k) / k

            # Compute recall@K
            recall_k = len(hit_set_at_k) / (all_relevants + 1e-12)
            # =======================================
            # =======================================
            vals_precision.append(precision_k)
            vals_recall.append(recall_k)
            if precision_k + recall_k > 0:
                vals_f1.append((2 * precision_k * recall_k) / (precision_k + recall_k))
            else:
                vals_f1.append(0)

        print(len(vals_precision)/len(all_relevant_indices_))
        results[key] = (np.mean(vals_precision), np.mean(vals_recall), np.mean(vals_f1), len(vals_precision)/len(all_relevant_indices_) )


    return results





########################################################################################################################
########################################################################################################################
########################################################################################################################
def dcg_score(y_true, y_score, k=10, gains="exponential"):

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

########################################################################################################################

def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    # num true labels in top k predictions / k
    # sortd_true = np.argsort(y_true)[:, ::-1]
    # topk_true = sortd_true[:, :k]


    ndcg_list = []
    for i in range(len(y_true)):
        all_relevants = sum((y_true[i] > 1e-05))
        if all_relevants == 0:
            continue
        best = dcg_score(y_true[i], y_true[i], k, gains)
        actual = dcg_score(y_true[i], y_score[i], k, gains)
        ndcg_list.append(actual / best)

    return np.mean(ndcg_list)
########################################################################################################################
########################################################################################################################
########################################################################################################################

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)

    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return [prec, rec, f1]



# y = np.asarray([[1,0,0,1,0,0,1],[0,1,0,1,0,1,0]])
#
# yhat_raw = np.asarray([[0.2,0,0,0.5,0,0,0.1],[0.6,0,0,0.4,0,0,0]])
# print (precision_at_k(yhat_raw, y, 1))
# print (precision_at_k(yhat_raw, y, 2))
# print (precision_at_k(yhat_raw, y, 3))
# print (precision_at_k(yhat_raw, y, 4))
# print (precision_at_k(yhat_raw, y, 5))