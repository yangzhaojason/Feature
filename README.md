# Multi-attribute feature extraction

The goal of this algorithm is to extract the characteristics of multi-dimensional (high-dimensional) attributes and select the series attributes with large attribute weights. The difficulty of the algorithm is to calculate the attribute weights of the corresponding columns, not the dimensionality reduction or eigenvalue decomposition alone(Order), so just using traditional SVD or PCA coordinate mapping does not apply to this scenario.

## Principal component analysis algorithm（PCA）

The covariance matrix is calculated using the PCA algorithm. The elements on the main diagonal of the covariance matrix are the variances (ie, energy) in each dimension, and the other elements are the covariances (ie, correlations) between the two dimensions, so It is based on the traditional algorithm - the largest theory of variance.

- [PCA Covariance Matrix Article](https://blog.csdn.net/makenothing/article/details/46390269)
- [PCA coding](https://blog.csdn.net/u012162613/article/details/42177327)

## Tree structure modeling（XGBoost）

XGBoost is a gradient lifting algorithm and residual decision tree. The basic idea is that a tree and a tree are gradually added to the model. When adding a CRAT decision tree, the overall effect (the objective function is Declining) has improved. A plurality of decision trees (a plurality of single weak classifiers) are used to form a combined classifier, and each leaf node is assigned a certain weight. In general, the importance score measures the value of the feature in the construction of the decision tree in the model. The more an attribute is used to build a decision tree in a model, the more important it is.

- [Gradient lifting algorithm how to calculate feature importance-coding](https://blog.csdn.net/waitingzby/article/details/81610495)

- [Python Sklearn implements two-class and multi-classification of xgboost](https://blog.csdn.net/ping550/article/details/79876298)

  Three methods of calculating the importance of features are provided in XGBoost：

  > ‘weight’ - the number of times a feature is used to split the data across all trees. 
  > ‘gain’ - the average gain of the feature when it is used in trees.
  > ‘cover’ - the average coverage of the feature when it is used in trees.

- [Feature importance calculation related code in XGBoost-get_score()](https://blog.csdn.net/zhangbaoanhadoop/article/details/81840656)

## Supplement：LightGBM

LightGBM is a method based on GBDT. For this type of tree-based model, the most time-consuming part is that when performing feature selection node splitting, it is necessary to traverse all possible partition points and calculate the information gain to find the optimal partition point. Although there are various algorithms to optimize this process, such as XGBoost, its efficiency and flexibility are not good enough when the feature dimension is high and the sample size is large. Therefore, the author of this paper proposed the LightGBM model, which greatly improved the computational efficiency. According to the paper, the training speed is 20 times faster than the GBDT. Therefore, the main problem to be solved by the LightGBM model is the problem of computational efficiency. While fast, it also guarantees the accuracy of the model, which is its biggest advantage.

## Code running

1) Install the xgboost library（support windows/Linux/macOS）

2) python Feature_extraction.py 

