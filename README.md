# Application-of-Clustering-Models

In this project, we will calculate the historic returns and volatility for the S&P 500 stocks and then proceed to use the K-Means clustering algorithm to divide the stocks into distinct groups based upon said returns and volatilities. Dividing stocks into groups with “similar characteristics” can help in portfolio construction to ensure we choose a universe of stocks with sufficient diversification between them.

The concept behind K-Means clustering is explained at https://en.wikipedia.org/wiki/K-means_clustering.

<b>Dataset Link: S&P 500 stocks</b>
<br>https://drive.google.com/file/d/1pP0Rr83ri0voscgr95-YnVCBv6BYV22w/view

Download the dataset from the above location and run the below script:

---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans,vq
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from math import sqrt
```
---

### Read data from the CSV file
```python
df = pd.read_csv('data_stocks.csv')
```

This gets up something resembling the following:

![Dataframe](https://github.com/siddharthalal/Project---Application-of-Clustering-Models/blob/master/dataframe.png?raw=true)

We can now start to analyse the data and begin our K-Means investigation.

Our first decision is to choose how many clusters do we actually want to separate the data into. Rather than make some arbitrary decision we can use an “Elbow Curve” to highlight the relationship between how many clusters we choose, and the Sum of Squared Errors (SSE) resulting from using that number of clusters.

We then plot this relationship to help us identify the optimal number of clusters to use – we would prefer a lower number of clusters, but also would prefer the SSE to be lower – so this trade off needs to be taken into account.

Lets run the code for our Elbow Curve plot.

```python
#Calculate percentage return and volatilities over the period
returns = df.pct_change().mean() * 109
returns = pd.DataFrame(returns)
returns.columns = ['Returns']
returns['Volatility'] = df.pct_change().std() * sqrt(109)
returns.head()
```

```python
#format the data as a numpy array to feed into the K-Means algorithm
X = returns
distorsions = []
for k in range(2, 20):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    distorsions.append(k_means.inertia_)
 
fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('Elbow curve')
```

![Elbow curve](https://github.com/siddharthalal/Project---Application-of-Clustering-Models/blob/master/Elbow%20Curve.png?raw=true)

So we can sort of see that once the number of clusters reaches 5 (on the bottom axis), the reduction in the SSE begins to slow down for each increase in cluster number. This would lead me to believe that the optimal number of clusters for this exercise lies around the 5 mark – so let’s use 5.

```python
# computing K-Means with K = 5 (5 clusters)
centroids,_ = kmeans(data,5)

# assign each sample to a cluster
idx,_ = vq(data,centroids)
 
# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()
```

This gives us the output:

![Cluster 1](https://github.com/siddharthalal/Project---Application-of-Clustering-Models/blob/master/Cluster%201.png?raw=true)

Ok, so it looks like we have an outlier in the data which is skewing the results and making it difficult to actually see what is going on for all the other stocks. Let’s delete the outlier from our data set and run this again.

```python
#identify the outlier
print(returns.idxmax())


#drop the relevant stock from our data#drop t 
returns.drop('NYSE.XRX',inplace=True)

#recreate data to feed into the algorithm
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T

# computing K-Means with K = 5 (5 clusters)
centroids,_ = kmeans(data,5)
# assign each sample to a cluster
idx,_ = vq(data,centroids)
 
# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()
```

This gets us a much clearer visual representation of the clusters as follows:

![Cluster 2](https://github.com/siddharthalal/Project---Application-of-Clustering-Models/blob/master/Cluster%202.png?raw=true)

Finally to get the details of which stock is actually in which cluster we can run the following line of code to carry out a list comprehension to create a list of tuples in the (Stock Name, Cluster Number) format:

```python
details = [(name,cluster) for name, cluster in zip(returns.index,idx)]

for detail in details:
    print(detail)
```

This will print out something resembling the below (I havn’t included all the results for brevity):

![Cluster 2](https://github.com/siddharthalal/Project---Application-of-Clustering-Models/blob/master/result.png?raw=true)

So there we have it, we now have a list of each of the stocks in the S&P 500, along with which one of 5 clusters they belong to with the clusters being defined by their return and volatility characteristics. We also have a visual representation of the clusters in chart format.
