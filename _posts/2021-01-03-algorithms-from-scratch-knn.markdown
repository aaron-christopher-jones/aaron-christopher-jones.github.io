---
title: "Algorithms from Scratch: KNN"
layout: post
date: 2021-01-03
image: 
headerImage: false
tag:
- algorithms from scratch
- classification
- k nearest neighbors
category: blog
author: aaron
description: 
---

## Algorithms from Scratch:
# K-Nearest Neighbors (KNN)

Despite virtually all statistical / machine learning algorithms being available as easily callable functions from open source libraries, I believe a strong working knowledge of the algorithms is still imperative. It has been a hobby of mine to code these algorithms from scratch to confirm and expand my working knowledge. I am going to start sharing some of this work in a series of blog posts. This first post will be on k-nearest neighbors.

K-nearest neighbors is an algorithm that can be used for both classification and regression, although it is probably most commonly used for classification. The flow of the algorithm is nearly identical for continuous and categorical dependent variables. The only difference is how the nearest neighbors are combined to produce the final prediction.

The process of k-nearest neighbors is:

1. Compute the distance between every observation in the training dataset and every observation in the prediction dataset. The distance metric used in my version below is the Minkowski distance of order p. Minkowski distance of order p is $D(X, Y) = \sqrt(\sum_{i=1}^{n}(x_{i} - y_{i})^{2})$ where each *i* is a different feature of the observation vector.
2. Select k observations from the training dataset for each observation in the prediction dataset. The k hyperparameter is either set in advance by the user or optimized through some search procedure. In this case, the k value is set to 5. The k observations are those with the minimum distances.
3. Aggregate the k observations to produce the final predicted value for the observation vector in the prediction dataset. In the case of this classification example, make the prediction the category value that occurs most often in the k observations of the training dataset. For a regression problem, the output value from the k observations might instead be the mean average.

Below is the coded from scratch version of the k-nearest neighbors algorithm and the results of running the algorithm on the breast cancer dataset.


```python
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as sp_dist
import seaborn
from statsmodels.graphics.mosaicplot import mosaic

plt.rcParams["figure.figsize"] = (60, 30)

seaborn.set(font_scale=3.5)
seaborn.set_style("whitegrid")
```


```python
cancer = (pd.read_csv("./BreastCancer.csv",
                      header=0,
                      names=["id", "diag", "radius", "texture", "perimeter", 
                             "area", "smoothness", "compactness", "concavity", 
                             "concave_points", "symmetry", "fractal_dimension"])
          .sample(frac=1)
          .dropna(axis=0)
          .drop(columns=["id"])
          .replace({"diag": {"M": 1, "B": 0}}))

nvalid = int(np.floor(cancer.shape[0] * 0.2))

train = cancer.head(n=-nvalid)
valid = cancer.tail(n=nvalid)

Xtrain = train.drop(columns=['diag']).to_numpy()
ytrain = train['diag'].values

Xvalid = valid.drop(columns=['diag']).to_numpy()
yvalid = valid['diag'].values
```

Below are the category percentages. The percentages are needed for understanding how the dataset is balanced, which in turn helps the interpretation of model quality / performance.


```python
list_targets = [("Total", cancer.diag.values), ("Train", ytrain), ("Valid", yvalid)]
print("Label order:   [0 1]")
print("Target: Counts, Percentages:")
for name_, array_ in list_targets:
    vals, cnts = np.unique(array_, return_counts=True)
    print("\t{0}: {1}, {2}".format(name_, cnts, np.around(cnts / len(array_), 2)))
```

    Label order:   [0 1]
    Target: Counts, Percentages:
    	Total: [357 210], [0.63 0.37]
    	Train: [288 166], [0.63 0.37]
    	Valid: [69 44], [0.61 0.39]


Here is the custom k-nearest neighbors algorithm. It is a relatively simple algorithm, but in many cases it is very useful and accurate while maintaining low computational requirements.


```python
def knn(X1, y1, X2, num_neighbors):
    """
    Execute k nearest neighbors algorithm for classification.

    Parameters
    ----------
    X1 : Dd-array
        covariate values, train dataset
    y1 : 1d-array
        response values, train dataset
    X2 : Dd-array
        covariate values, valid dataset
    num_neighbors : integer
        number of neighbors, k value

    Returns
    -------
    array
        response values, valid dataset
        number of values should equal number of rows in X2
    """
    dists = sp_dist.cdist(X2, X1, "minkowski", p=2)
    min_indices = [dists[i,:].argsort()[:num_neighbors] for i in range(X2.shape[0])]
    y2 = [np.argmax(np.bincount(y1[i])) for i in min_indices]
    return np.array(y2)
```


```python
ypred = knn(X1=Xtrain, y1=ytrain, X2=Xvalid, num_neighbors=5)
```

The confusion matrix combined with the category percentages show that the k-nearest neighbors algorithm is way outperforming any benchmarking approach like guessing only one category or guessing randomly.


```python
output = pd.DataFrame({"Actual": yvalid, "Predict": ypred})
confusion = pd.crosstab(output.Actual, output.Predict)
confusion / len(yvalid)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predict</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.566372</td>
      <td>0.044248</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.088496</td>
      <td>0.300885</td>
    </tr>
  </tbody>
</table>
</div>



This mosaic plot is a nice visual representation of the confusion matrix.


```python
outputmosaic = pd.DataFrame({
    "Actual": ["Actual=0", "Actual=0", "Actual=1", "Actual=1"], 
    "Predict": ["Predict=0", "Predict=1", "Predict=0", "Predict=1"],
    "Percs": [0.610619, 0.035398, 0.070796, 0.283186]
}).set_index(["Actual", "Predict"])
mosaic(outputmosaic.Percs, gap=0.01, title='Mosaic Confusion Matrix', axes_label=False)
plt.show()
```


![png](assets/images/algos_from_scratch_knn/output_12_0.png)


The below spider plot describes the model performance in 6 different metrics. The high values for accuracy, F1, sensitivity, and specificity along with the low values for false negative rate and false positive rate confirm the quality of the model. It is possible further model refinement (i.e., adjusting the k value) could improve results, but this model is performing decently well. Again, model performance isn't the main concern here. The goal of these blog posts is digging into the internals of the algorithm, but it is good to confirm that the custom version of the algorithm works!


```python
tn = confusion.iloc[0, 0]
tp = confusion.iloc[1, 1]
fp = confusion.iloc[0, 1]
fn = confusion.iloc[1, 0]

metric = ["Accurary", "F1", "Sensitivity", "Specificity", 
          "False Negative Rate", "False Positive Rate"]
N = len(metric)
score = [(tp + tn) / (tp + fp + fn + tn), tp / (tp + 0.5 * (fp + fn)), 
         tp / (tp + fn), tn / (tn + fp), fn / (fn + tp), fp / (fp + tn)]
score += score[:1]

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.polar(angles, score)
plt.xticks(angles[:-1], metric)
plt.yticks(list(np.linspace(0.2, 1, 5)), color="grey", size=35)
plt.ylim(0, 1)
plt.title("Model Performance")
plt.show()
```


![png](assets/images/algos_from_scratch_knn/output_14_0.png)


