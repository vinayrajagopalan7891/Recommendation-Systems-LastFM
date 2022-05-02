import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import surprise

class ContentBased:

    def __init__(self, NN):
        self.NN = NN
        self.fitted = {"content":False, "ratings":False}
        
    def fit(self, content_data):

        self.items = content_data.index.values
        self.item_dim = len(self.items)
        # check for duplicate items
        assert (len(self.items) == len(set(self.items))), "Duplicate items in content data!"

        # compute similarity
        self.matrix = cosine_similarity(content_data.values)
        np.fill_diagonal(self.matrix, 0)
        
        self.matrixNN = self.matrix.copy()

        # filter similarity matrix for NN nearest neighbors (constraint: non-negative similarity)
        for i in range(self.item_dim):
            crit_val = max(-np.sort(-self.matrix[i])[self.NN-1], 0)
            self.matrixNN[i][self.matrixNN[i] < crit_val] = 0.0
        
        self.fitted["content"] = True

    # helper -> transform surprise.trainset.Trainset to pd.DataFrame
    def _trainset2list(self, trainset):
        return pd.DataFrame([(trainset.to_raw_uid(u), trainset.to_raw_iid(i), r) for (u, i, r) in trainset.all_ratings()], columns=["user", "item", "rating"])

    def fit_ratings(self, df):

        if not self.fitted["content"]:
            raise Exception("Fit model on content data!")

        if isinstance(df, surprise.trainset.Trainset):
            df = self._trainset2list(df)
        
        # fix unknown items
        unknown_items = list(set(df["item"]) - set(self.items))
        if len(unknown_items) > 0:
            print(f"Warning {len(unknown_items)} items are not included in content data: {unknown_items}")
        df = df[df["item"].isin(self.items)].reset_index(drop=True)

        # store user data
        self.users = np.unique(df["user"])
        self.user_dim = len(self.users)

        # fix missing items
        missing_items = list(set(self.items) - set(df["item"]))
        if len(missing_items) > 0: 
            fix_df = pd.DataFrame([{"user":np.nan, "item":i, "rating":np.nan} for i in missing_items])
            df = df.append(fix_df).reset_index(drop=True)

        # pivot 
        df_pivot = df.pivot_table(index='user', values='rating', columns='item', dropna=False).reindex(self.users)

        # row-wise (user) average
        self.user_avg = np.array(np.mean(df_pivot, axis=1))
        self.global_mean = np.mean(self.user_avg)

        # center ratings
        df_pivot = df_pivot.sub(self.user_avg, axis=0).fillna(0)

        # predict ratings for each item 
        denom = self.matrixNN.sum(axis=0) # column sums
        self.prediction = (np.matmul(df_pivot.values, self.matrixNN) / denom) + self.user_avg[:,np.newaxis]

        # replace NA values with mean
        # prediction[np.isnan(prediction)] = self.global_mean

        self.fitted["ratings"] = True
    
    # get predicted value for user-item combination
    def predict(self, user, item, r_ui=None):
        details = {"was_impossible":False}

        # check whether user and item are unknown -> default = global average rating
        if self.knows_user(user) & self.knows_item(item):

            # convert user & item in internal ids
            iid = np.where(self.items == item)[0].item()
            uid = np.where(self.users == user)[0].item()

            # inference prediction
            est = self.prediction[uid, iid]
            
            if np.isnan(est): 
                est = self.global_mean
                details["was_impossible"] = True
            return surprise.Prediction(user, item, r_ui, est, details)
        
        else:
            details["was_impossible"] = True
            details["reason"] = "User or item unknown"
            return surprise.Prediction(user, item, r_ui, self.global_mean, details)

    # predict entire testset
    def test(self, testset):
        if not self.fitted["ratings"]:
            raise Exception("Fit model on ratings data!")
        return [self.predict(user=u,item=i,r_ui=r) for (u,i,r) in testset]

    def knows_user(self, user):
        return user in self.users   

    def knows_item(self, item):
        return item in self.items        

    # get topn most similar items 
    def get_most_similar(self, item, topn=5):

        # get iid
        if self.knows_item(item):
            iid = np.where(self.items == item)[0].item()
        else:
            raise Exception(f"Item {item} unknown ...")
        
        list_iids = (-self.matrix[iid]).argsort()[:topn]
        return self.items[list_iids]

    def get_similarities(self):
        print('Cosine similarities shape: ({}, {}) items x items'.format(self.item_dim, self.item_dim))
        return self.matrix