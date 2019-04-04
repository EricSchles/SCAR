import pandas as pd
from sklearn import cluster
from sklearn import neighbors
from sklearn import metrics
from sklearn import tree
from sklearn import svm
from sklearn import linear_model

class SCAR:
    def __init__(self):
        self.models = {}

    # consider test train split
    def cluster_training(self, max_k):
        best_k = 2
        lowest_mse = 1000000000
        for k in range(2, max_k):
            knn = neighbors.KNeighborsRegressor(n_neighbors=k)
            knn.fit(self.X, self.y)
            y_pred = knn.predict(self.X)
            mse = metrics.mean_squared_error(self.y, y_pred)
            if mse < lowest_mse:
                best_k = k
                lowest_mse = mse
        self.k_means = cluster.KMeans(n_clusters=best_k)
        self.k_means.fit(self.X)
        self.df["cluster"] = self.k_means.predict(self.X)
        self.num_clusters = best_k
        
    def label_cluster(self, new_data):
        new_data["cluster"] = self.k_means.predict(new_data)
        return new_data

    def _fit_models(self, model_obj):
        for cluster in range(self.num_clusters):
            model = model_obj()
            X = self.X[self.X["cluster"] == cluster]
            y = self.y.loc[X.index]
            model.fit(X, y)
            self.models[cluster] = model

    def fit_decision_trees(self):
        self._fit_models(tree.DecisionTreeRegressor)
        
    def fit_svms(self):
        self._fit_models(svm.SVR)

    def fit_linears(self):
        self._fit_models(linear_model.LinearRegression)
        
    def fit(self, max_k, kind=""):
        if kind == "decision_tree":
            self.cluster_training(max_k)
            self.fit_decision_trees()
        if kind == "svm":
            self.cluster_training(max_k)
            self.fit_svms()
        if kind == "linear":
            self.cluster_training(max_k)
            self.fit_linears()
            
    def predict(self, X):
        labeled_data = self.label_cluster(X)
        total_predictions = []
        for cluster in labeled_data["cluster"].unique():
            clustered_X = labeled_data[labeled_data["cluster"] == cluster]
            predictions = models[cluster].predict(clustered_X)
            predictions = pd.Series(predictions)
            predictions.index = clustered_X.index
            total_predictions.append(predictions)
        return pd.concat(total_predictions)
