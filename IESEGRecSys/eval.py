import pandas as pd 
import numpy as np 
from numpy.linalg import norm
from sklearn.metrics import ndcg_score, roc_auc_score, roc_curve
import surprise

def simil_cosine(a,b):
    return np.dot(a, b)/(norm(a)*norm(b))

def pearson_corr(a,b):
    return np.corrcoef(a, b)[0,1]

# compute RMSE
def rmse(pred, real):
    return np.sqrt(((pred - real) ** 2).mean())

# compute MAE
def mae(pred, real):
    return np.absolute(np.subtract(real, pred)).mean()

# check if input is pandas DataFrame
def isDataFrame(x):
    return isinstance(x, pd.DataFrame)

# convert predictions from pandas DataFrame to surprise.Prediction
def df2surprise(df):
    return [surprise.Prediction(*i) for i in df.itertuples(index=False)]

# Wrapper for accuracy metrics
def prediction_metrics(prediction, excl_impossible=False):

    if isDataFrame(prediction):
        prediction = df2surprise(prediction)

    pred = []
    real = [] 

    res = {}
    if excl_impossible:
        if all(isinstance(i, surprise.Prediction) for i in pred):

            # filter impossible prediction (np.array([True, False, True, ...])) 
            for i in prediction:
                if not i.details["was_impossible"]:
                    pred.append(i.est)
                    real.append(i.r_ui) 
            print(f'Excluded {len(prediction) - len(pred)} ({len(prediction)}) samples. {len(pred)} remaining ...')
        else:
            raise Exception(f"Argument 'prediction' not of type -> List[surprise.Prediction]")
    else:

        for i in prediction:
            pred.append(i.est)
            real.append(i.r_ui)

    # compute accuracy metrics
    res['RMSE'] = [rmse(np.array(pred), np.array(real))]
    res['MAE'] = [mae(np.array(pred), np.array(real))]
    
    # return pd.DataFrame
    return pd.DataFrame(res, index=['value']).T


def classification_metrics(prediction, threshold, topn=False, excl_impossible=False):

    if isDataFrame(prediction):
        prediction = df2surprise(prediction)

    list_metrics = ['Recall', 'Precision', 'F1']

    if topn:
        print('Warning: TopN classification not recommended to use ...')

        if excl_impossible:
            prediction = [i for i in prediction if not i.details["was_impossible"]]

        df_pred_sort = {}
        df_real_sort = {}
        df_pred = pd.DataFrame(prediction)

        # sort user preferences by rating
        for user in df_pred['uid'].unique():
            # predicted values
            plist = list(df_pred.loc[df_pred['uid']==user,:].sort_values('est', ascending=False)['iid'])
            df_pred_sort[user] = [plist]
            
            # real values
            rlist = list(df_pred.loc[df_pred['uid']==user,:].sort_values('r_ui', ascending=False)['iid'])
            df_real_sort[user] = [rlist]

        df_pred_sort = pd.DataFrame(df_pred_sort, index=['item']).T
        df_real_sort = pd.DataFrame(df_real_sort, index=['item']).T
        
        # check for consistency in recommended items
        if (len(set([len(i[:threshold]) for i in df_pred_sort["item"]])) > 2):
            print('Warning: Missing items in TopN predictions may lead to imprecise metrics ...')

        list_tp = []
        list_fp = []
        list_fn = []

        # compute overlap recommendations and true user preferences
        for pred, real in zip(df_pred_sort["item"], df_real_sort["item"]):
            common = len(np.intersect1d(pred[:threshold], real[:threshold]))
            list_tp.append(common)
            list_fp.append(len(pred[:threshold]) - common)
            list_fn.append(len(real[:threshold]) - common)
        TP = np.sum(list_tp)
        FP = np.sum(list_fp)
        FN = np.sum(list_fn)

        list_metrics = [f'Recall@{threshold}', f'Precision@{threshold}', f'F1@{threshold}']
    
    else:
        pred = []
        real = [] 

        if excl_impossible:
            if all(isinstance(i, surprise.Prediction) for i in pred):

                # filter impossible prediction (np.array([True, False, True, ...])) 
                for i in prediction:
                    if not i.details["was_impossible"]:
                        pred.append(i.est)
                        real.append(i.r_ui) 
                print(f'Excluded {len(prediction) - len(pred)} ({len(prediction)}) samples. {len(pred)} remaining ...')
            else:
                raise Exception(f"Argument 'prediction' not of type -> List[surprise.Prediction]")
                
        else:
            for i in prediction:
                pred.append(i.est)
                real.append(i.r_ui)

        pred = np.array(pred)
        real = np.array(real)
        
        # compute confusion matrix
        TP = np.sum((pred >= threshold) & (real >= threshold))
        FP = np.sum((pred >= threshold) & (real < threshold))
        FN = np.sum((pred < threshold) & (real >= threshold))
        TN = np.sum((pred < threshold) & (real < threshold))


    # compute Recall, Precision, F1
    recall    = TP / (TP+FN)
    precision = TP / (TP+FP)
    f1        = (2*precision*recall) / (precision+recall)

    df = pd.DataFrame([recall, precision, f1], index=list_metrics, columns=['value'])

    return df

def ndcg(prediction, real, topn):
    res = []
    for rl, pred in zip(real, prediction):
        # overlap between predictions and real preferences
        common = set(rl).intersection(set(pred))
        # extract ranking of common items
        rl_rank = [rl.index(x) for x in common]
        pred_rank = [pred.index(x) for x in common]

        # compute NDCG
        if len(rl_rank) == 1: 
            if rl_rank==pred_rank: res.append(1.0)
            else: res.append(0.0)
        else:
            res.append(ndcg_score([rl_rank[:topn]], [pred_rank[:topn]]))
    return np.mean(res)

# Removed - Possible to compute in theory but practical value is very limited for RecSys
# def auc(prediction, real, threshold, plot=True, excl_impossible=False, details=''):

#     if excl_impossible:
#         # check shapes
#         if len(details) != len(prediction):
#             raise Exception('Shapes do not match: prediction={}, details={}'.format(len(prediction), len(details)))
#         else:
#             # filter impossible prediction (np.array([True, False, True, ...])) 
#             arr_filt = np.array([False if i['was_impossible'] else True for i in details])
#             print('Excluded {} ({}) samples. {} remaining ...'.format(len(arr_filt) - np.sum(arr_filt),len(arr_filt), np.sum(arr_filt)))

#             # filter predictions and real values
#             prediction_filt, real_filt = np.array(prediction)[arr_filt], np.array(real)[arr_filt]
#             pred = [1 if i >= threshold else 0 for i in prediction_filt]
#             rl = [1 if i >= threshold else 0 for i in real_filt]

#     else:
#         pred = [1 if i >= threshold else 0 for i in prediction]
#         rl = [1 if i >= threshold else 0 for i in real]
    
#     score = roc_auc_score(rl, pred)

#     if plot:
#         fpr, tpr, _ = roc_curve(rl, pred)
#         plt.figure()
#         plt.plot(fpr, tpr, color='darkorange',
#                 lw=2, label='ROC curve (area = {})'.format(round(score,3)))
#         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.0])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.legend(loc="lower right")
#         plt.show()
    
#     # Return AUC score
#     return score


def ranking_metrics(prediction, threshold, excl_impossible=False):

    if isDataFrame(prediction):
        prediction = df2surprise(prediction)

    if excl_impossible:
        prediction = [i for i in prediction if not i.details["was_impossible"]]

    df_pred_sort = {}
    df_real_sort = {}
    df_pred = pd.DataFrame(prediction)

    # sort user preferences by rating
    for user in df_pred["uid"].unique():
        # predicted values
        plist = list(df_pred.loc[df_pred["uid"]==user,:].sort_values("est", ascending=False)["iid"])
        df_pred_sort[user] = [plist]
        
        # real values
        rlist = list(df_pred.loc[df_pred["uid"]==user,:].sort_values("r_ui", ascending=False)["iid"])
        df_real_sort[user] = [rlist]

    df_pred_sort = pd.DataFrame(df_pred_sort, index=["iid"]).T
    df_real_sort = pd.DataFrame(df_real_sort, index=["iid"]).T

    res = {}

    res[f'NDCG@{threshold}'] = ndcg(df_real_sort["iid"], df_pred_sort["iid"], threshold)

    return pd.DataFrame(res, index=['value']).T

def evaluate(prediction, topn, rating_cutoff, excl_impossible=False):
    if isDataFrame(prediction):
        prediction = df2surprise(prediction)
    return prediction_metrics(prediction, excl_impossible=excl_impossible).append(classification_metrics(prediction, rating_cutoff, excl_impossible=excl_impossible)).append(ranking_metrics(prediction, topn, excl_impossible=excl_impossible))

