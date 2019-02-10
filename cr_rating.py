from pyspark import SparkContext
from pyspark.sql import HiveContext
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from numpy.matlib import repmat
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy import cluster
#import matplotlib.pylab as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv
from pyspark.sql.types import *

sc = SparkContext()
sqlc = HiveContext(sc)

#dataset=pd.read_csv('F:\Credit Rating\cr_rating.csv') ## Read from CSV
dataset=sqlc.sql("select * from credit_rating_input").toPandas()

def sci_minmax(X):
    minmax_scale = MinMaxScaler(feature_range=(0, 1), copy=True)
    return minmax_scale.fit_transform(X)

dataTBNorm=pd.DataFrame(dataset[["feature1","feature2","feature3","feature4","feature5"]])
norm_data=sci_minmax(dataTBNorm)
norm_data[:,1]=1-norm_data[:,1]

##############Elbow curve method to identify initial number of clusters##########################
clusters=range(1,10)
meandist=[]
for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(norm_data)
    clusassign=model.predict(norm_data)
    meandist.append(sum(np.min(cdist(norm_data, model.cluster_centers_, 'euclidean'), axis=1))
    / norm_data.shape[0])

nPoints = len(meandist)
allCoord = np.vstack((range(nPoints), meandist)).T

np.array([range(nPoints), meandist])
firstPoint = allCoord[0]

lineVec = allCoord[-1] - allCoord[0]
lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
vecFromFirst = allCoord - firstPoint
scalarProduct = np.sum(vecFromFirst * repmat(lineVecNorm, nPoints, 1), axis=1)
vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
vecToLine = vecFromFirst - vecFromFirstParallel
distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
idxOfBestPoint = np.argmax(distToLine)
numberOfClus= idxOfBestPoint+1

#############################Final K-means#############################################
kmeans = KMeans(n_clusters=10, random_state=0).fit(norm_data)
clus=np.reshape(kmeans.labels_,(len(norm_data),1))
final_data=np.hstack((dataset[["mobile","feature1","feature2","feature3","feature4","feature5"]], clus))
final_data=pd.DataFrame(final_data)

final_data.columns=["mobile","mobile","feature1","feature2","feature3","feature4","feature5","cluster_overall"]

cls_centers=kmeans.cluster_centers_
norm_data_centers=sci_minmax(cls_centers)

cls_label=kmeans.labels_.tolist()

rows=kmeans.n_clusters
col=3
cluster_out=[[0 for x in range(col)] for x in range(rows)]

for i in range(0,kmeans.n_clusters):
        ci=norm_data_centers[i][0]*.25+norm_data_centers[i][1]*.15+norm_data_centers[i][2]*.15+norm_data_centers[i][3]*.2+norm_data_centers[i][4]*.1+norm_data_centers[i][5]*.1+norm_data_centers[i][6]*.05
        cluster_out[i][0]=i
        cluster_out[i][1]=ci
        cluster_out[i][2]=cls_label.count(i)

cluster_out=(sorted(cluster_out,key=lambda x:x[1]))

final_data["cluster_overall"]=final_data["cluster_overall"].astype(str)

rows_final=kmeans.n_clusters
col_final=3
cluster_out_final=[[0 for x in range(col)] for x in range(rows)]

for i in range(0,len(cluster_out)):
    final_data["cluster_overall"] = np.where(final_data["cluster_overall"] == str(cluster_out[i][0]), '0'+str(i+1) , final_data["cluster_overall"])
    print("clus ",i+1, "=", cluster_out[i][1], "count =", cluster_out[i][2])
    cluster_out_final[i][0]="clus "+str(i+1)
    cluster_out_final[i][1]=cluster_out[i][1]
    cluster_out_final[i][2]=cluster_out[i][2]

with open('/home/CREDIT_RATING/overall_clus_matrix.csv', "w") as output:
    writer=csv.writer(output,lineterminator='\n')
    writer.writerows(cluster_out_final)

###########################Individual Clustering#########################################
individual_data=dataset[["mobile","feature1","feature2","feature3","feature4","feature5"]]
norm_Ind_data=sci_minmax(individual_data)
norm_Ind_data[:,2]=1-norm_Ind_data[:,2]
norm_Ind_data=pd.DataFrame(norm_Ind_data)

clus_centers_ind=[]

for column in norm_Ind_data:
    clusters_ind = range(1, 10)
    meandist_ind = []
    for k in clusters_ind:
        model_ind = KMeans(n_clusters=k)
        model_ind.fit(norm_Ind_data[[column]])
        clusassign_ind = model_ind.predict(norm_Ind_data[[column]])
        meandist_ind.append(sum(np.min(cdist(norm_Ind_data[[column]], model_ind.cluster_centers_, 'euclidean'), axis=1))
                        / norm_Ind_data[[column]].shape[0])

    nPoints_ind = len(meandist_ind)
    allCoord_ind = np.vstack((range(nPoints_ind), meandist_ind)).T

    np.array([range(nPoints_ind), meandist_ind])
    firstPoint_ind = allCoord_ind[0]

    lineVec_ind = allCoord_ind[-1] - allCoord_ind[0]
    lineVecNorm_ind = lineVec_ind / np.sqrt(np.sum(lineVec_ind ** 2))
    vecFromFirst_ind = allCoord_ind - firstPoint_ind
    scalarProduct_ind = np.sum(vecFromFirst_ind * repmat(lineVecNorm_ind, nPoints_ind, 1), axis=1)
    vecFromFirstParallel_ind = np.outer(scalarProduct_ind, lineVecNorm_ind)
    vecToLine_ind = vecFromFirst_ind - vecFromFirstParallel_ind
    distToLine_ind = np.sqrt(np.sum(vecToLine_ind ** 2, axis=1))
    idxOfBestPoint_ind = np.argmax(distToLine_ind)
    numberOfClus_ind = idxOfBestPoint_ind + 1

    kmeans_ind = KMeans(n_clusters=10, random_state=0).fit(norm_Ind_data[[column]])
    clus_ind = np.reshape(kmeans_ind.labels_, (len(norm_Ind_data[[column]]), 1))
    final_data = np.hstack((final_data, clus_ind))
    clus_centers_ind.append(kmeans_ind.cluster_centers_)

final_data = pd.DataFrame(final_data)
final_data.columns = ["mobile","mobile","feature1","feature2","feature3","feature4","feature5","cluster_feature1", "cluster_feature2", "cluster_feature3","cluster_feature4",
                      "cluster_feature5"]

#############################For Feature1###############################################
money_ind=pd.DataFrame(clus_centers_ind[0])
money_ind['clus']=range(0,len(money_ind))
money_ind_sort=money_ind.sort_values(by=0)
final_data["cluster_feature1"]=final_data["cluster_feature1"].astype(str)
print("For feature1 Cluster")
for i in range(0,len(money_ind_sort)):
    final_data["cluster_feature"] = np.where(final_data["cluster_feature"] == str(money_ind_sort['clus'].iloc[i]), '0'+str(i+1) , final_data["cluster_feature"])
    print("clus ",i+1, "=", money_ind_sort[0].iloc[i])

###########################For feature2#################################################
freq_ind=pd.DataFrame(clus_centers_ind[1])
freq_ind['clus']=range(0,len(freq_ind))
freq_ind_sort=freq_ind.sort_values(by=0)
final_data["cluster_feature2"]=final_data["cluster_feature2"].astype(str)
print("For feature2 Cluster")
for i in range(0,len(freq_ind_sort)):
    final_data["cluster_feature2"] = np.where(final_data["cluster_feature2"] == str(freq_ind_sort['clus'].iloc[i]), '0'+str(i+1) , final_data["cluster_feature"])
    print("clus ",i+1, "=", freq_ind_sort[0].iloc[i])

###############################For feature3##############################################
recen_ind=pd.DataFrame(clus_centers_ind[2])
recen_ind['clus']=range(0,len(recen_ind))
recen_ind_sort=recen_ind.sort_values(by=0)
final_data["cluster_feature3"]=final_data["cluster_feature3"].astype(str)
print("For feature3 Cluster")
for i in range(0,len(recen_ind_sort)):
    final_data["cluster_feature3"] = np.where(final_data["cluster_feature3"] == str(recen_ind_sort['clus'].iloc[i]), '0'+str(i+1) , final_data["cluster_feature3"])
    print("clus ",i+1, "=", recen_ind_sort[0].iloc[i])

#############################For feature4#######################################
ontime_ind=pd.DataFrame(clus_centers_ind[3])
ontime_ind['clus']=range(0,len(ontime_ind))
ontime_ind_sort=ontime_ind.sort_values(by=0)
final_data["cluster_feature4"]=final_data["cluster_feature4"].astype(str)
print("For feature4 Cluster")
for i in range(0,len(ontime_ind_sort)):
    final_data["cluster_feature4"] = np.where(final_data["cluster_feature4"] == str(ontime_ind_sort['clus'].iloc[i]), '0'+str(i+1) , final_data["cluster_feature4"])
    print("clus ",i+1, "=", ontime_ind_sort[0].iloc[i])

########################For feature5#########################################
contpay_ind=pd.DataFrame(clus_centers_ind[4])
contpay_ind['clus']=range(0,len(contpay_ind))
contpay_ind_sort=contpay_ind.sort_values(by=0)
final_data["cluster_feature5"]=final_data["cluster_feature5"].astype(str)
print("For feature5 Cluster")
for i in range(0,len(contpay_ind_sort)):
    final_data["cluster_feature"] = np.where(final_data["cluster_feature"] == str(contpay_ind_sort['clus'].iloc[i]), '0'+str(i+1) , final_data["cluster_feature"])
    print("clus ",i+1, "=", contpay_ind_sort[0].iloc[i])


final_data.index.name=None
final_data=pd.DataFrame(final_data)
print(final_data)
final_data.to_csv("/home/CREDIT_RATING/output_final.csv",index = False)

schema = StructType([StructField('feature1', StringType()),
                     StructField('feature2', DoubleType()),
                     StructField('feature3', DoubleType()),
                     StructField('feature4', DoubleType()),
                     StructField('feature5', DoubleType()),
                     StructField('cluster_overall', StringType()),
                     StructField('cluster_feature1', StringType()),
                     StructField('cluster_feature2', StringType()),
                     StructField('cluster_feature3', StringType()),
                     StructField('cluster_feature4', StringType()),
                     StructField('cluster_feature5', StringType())
                    ])

final_data=sqlc.createDataFrame(final_data,samplingRatio=0.2,schema=schema)
final_data.write.save(path='/source/credit_rating/input', source='parquet', mode='overwrite')