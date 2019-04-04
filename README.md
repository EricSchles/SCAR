# Similarity Clustering And Regression - SCAR

The goal of scar is to build a highly interpretability model that scales.  The way this is accomplished is by first clustering the data into similar points and then learning a distribution on each of the clusters.  From there, the right generalized least squares is applied to the data and a traditional model is learned.  When regressing the appropriate model is called based on which cluster it belongs to.  For highly non-linear that follows no distribution a tree based model is used instead.

## Psuedo Code

Training:

1. Cluster the data based on mean squared error
2. train a linear model on each of the clusters

Prediction:

1. Run a similarity measure against all new data points, label them according to which cluster they belong to.  
2. Draw predictions from the appropriate model.

