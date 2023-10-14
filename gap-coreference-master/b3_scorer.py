from collections import Counter
def B3_0(mentions, true_clusters, pred_clusters):
  '''
  Calculates precision, recall, and optionally F1 for the  B3(0) metric,
  based on formulation in https://aclanthology.org/W10-4305.pdf
  '''

  precision_scores = []
  recall_scores = []
  f1_scores = []

  for mention in mentions:
    precision = 0
    recall = 0

    # finding key and response clusters to look at (first cluster to come up that contains current mention)
    mention_key_cluster = None
    for cluster in true_clusters:
      if mention in cluster:
        mention_key_cluster = cluster
        break
    assert mention_key_cluster, "At least one true cluster must contain mention!"

    mention_pred_cluster = None
    for cluster in pred_clusters:
      if mention in cluster:
        mention_response_cluster = cluster
        break
    
    intersection_key_response = list((Counter(mention_key_cluster) & Counter(mention_response_cluster)).elements())
    overlap_count = len(intersection_key_response)

    # response cluster could be empty if mention was not predicted for any cluster (twinless mention); in this case precision and recall both at 0
    if mention_response_cluster:
      precision = overlap_count / len(mention_response_cluster)
      recall = overlap_count / len(mention_key_cluster)
    
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append((2*precision*recall)/(precision+recall))
  
  return precision_scores, recall_scores, f1_scores