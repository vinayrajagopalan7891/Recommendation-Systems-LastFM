from collections import defaultdict
import pandas as pd

# Adapt the get_top_n function from the surprise package
def get_top_n(predictions, n=15, only_new=False, train='', user_col='user', item_col='item'):
    """Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 15.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """
    top_n = defaultdict(list)

    # recommend only new items
    if only_new:
        if isinstance(train, pd.DataFrame):
            tagg = train.groupby(user_col).agg({item_col:lambda x: list(x)}).reset_index()
            tdict = dict(zip(tagg.user, tagg.item))
        else:
            raise Exception('ArgumentError: Train is not pd.DataFrame.')

        # Map the predictions to each user.
        for uid, iid, true_r, est, _ in predictions:
            if (uid in tdict.keys()) and (iid not in tdict[uid]):
                top_n[uid].append((iid, est))

    else:
        # Map the predictions to each user.
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    # topn to df
    df_topn = pd.DataFrame([(key, [v[0] for v in val]) for key, val in top_n.items()], columns=['user', 'item'])
    return df_topn

def predict_user_topn(model, train, user, topk=15, item_col='item'):
    """
    Returns topk items for specified user.
    Return type: list

    Args[model, train, user, topk, item_col]
    model:      fitted model (surprise package)
    train:      train set used to fit model
    user:       user id
    topk:       topk items to return
    item_col:   column containing item ids 
    """

    # intermediate result dict 
    res = {item_col:[], 'pred':[], 'detail':[]}
    
    # iter through all items contained in train set
    for item in set(train[item_col]) :
        uid, iid, true_r, est, detail = model.predict(user,item)
        if detail['was_impossible']: continue
        # save to result dict
        res[item_col].append(item)
        res['pred'].append(est)
        res['detail'].append(detail)

    return list(pd.DataFrame(res).sort_values('pred', ascending=False)[:topk][item_col])