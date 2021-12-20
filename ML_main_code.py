#!/usr/bin/env python
# coding: utf-8

# In[16]:


'''@author: harshmishra
   Detecting exoplanets using Machine Learning
'''
#Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, normalize 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,plot_confusion_matrix,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy import ndimage
from sklearn import  metrics
from sklearn.metrics import roc_curve,roc_auc_score,plot_roc_curve
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


train = pd.read_csv('exoTrain.csv')
test = pd.read_csv('exoTest.csv')
print(train.shape)
print(test.shape)


# In[18]:


#To check data imbalance
train['LABEL'].value_counts().plot(kind = 'bar', title = 'Class Distributions \n (0: Not Exoplanet || 1: Exoplanet)', rot=0)
# PROBLEM OF IMBALANCED CLASS -RUNNING ANY ALGORITHM WILL RESULT IN A ACCURACY MEASURE THAT IS HIGHLY INFLUENCED BY DOMINATING CLASS


# In[3]:


def stats_plots(df):
    means = df.mean(axis=1)
    medians = df.median(axis=1)
    std = df.std(axis=1)
    maxval = df.max(axis=1)
    minval = df.min(axis=1)
    skew = df.skew(axis=1)
    
    fig = plt.figure(figsize=(26,26))
    
    ax = fig.add_subplot(231)
    ax.hist(means,alpha=0.8,bins=100)
    ax.set_xlabel('Mean Intensity')
    ax.set_ylabel('Num. of Stars')
    
    ax = fig.add_subplot(232)
    ax.hist(medians,alpha=0.8,bins=100)
    ax.set_xlabel('Median Intensity')
    ax.set_ylabel('Num. of Stars')
    
    ax = fig.add_subplot(233)
    ax.hist(std,alpha=0.8,bins=100)
    ax.set_xlabel('Intensity Standard Deviation')
    ax.set_ylabel('Num. of Stars')
    
    ax = fig.add_subplot(234)
    ax.hist(maxval,alpha=0.8,bins=100)
    ax.set_xlabel('Maximum Intensity')
    ax.set_ylabel('Num. of Stars')
    
    ax = fig.add_subplot(235)
    ax.hist(minval,alpha=0.8,bins=100)
    ax.set_xlabel('Minimum Intensity')
    ax.set_ylabel('Num. of Stars')
    
    ax = fig.add_subplot(236)
    ax.hist(skew,alpha=0.8,bins=100)
    ax.set_xlabel('Intensity Skewness')
    ax.set_ylabel('Num. of Stars')


# In[4]:


stats_plots(test)
plt.show()


# In[5]:


#Scatter plots
#STARS WITH EXOPLANETS

fig = plt.figure(figsize=(15,40))
for i in range(12):
    ax = fig.add_subplot(14,4,i+1)
    ax.scatter(np.arange(3197),train[train['LABEL'] == 2].iloc[i,1:],s=1)

#STARS WITHOUT EXOPLANETS
fig = plt.figure(figsize=(15,40))
for i in range(12):
    ax = fig.add_subplot(14,4,i+1)
    ax.scatter(np.arange(3197),train[train['LABEL']==1].iloc[i,1:],s=1)    


# In[6]:


#STARS WITHOUT EXOPLANETS
fig = plt.figure(figsize=(15,40))
for i in range(12):
    ax = fig.add_subplot(14,4,i+1)
    ax.scatter(np.arange(3197),train[train['LABEL']==1].iloc[i,1:],s=1)


# In[19]:


#Plotting histograms
#STARS WITH EXOPLANETS
fig = plt.figure(figsize=(15,40))
for i in range(12):
    ax = fig.add_subplot(14,4,i+1)
    train[train['LABEL']==2].iloc[i,1:].hist(bins=50)
    
#STARS WITHOUT EXOPLANETS
fig = plt.figure(figsize=(15,40))
for i in range(12):
    ax = fig.add_subplot(14,4,i+1)
    train[train['LABEL']==1].iloc[i,1:].hist(bins=50)    


# In[ ]:


#Visualizing the 6 rercords
plt.figure(figsize=(25,10))
plt.title('Distribution of flux values', fontsize=15)
plt.xlabel('Flux values')
plt.ylabel('Flux intensity')
plt.plot(train.iloc[0,])
plt.plot(train.iloc[1,])
plt.plot(train.iloc[2,])
plt.plot(train.iloc[3,])
plt.plot(train.iloc[4,])
plt.plot(train.iloc[5,])

plt.legend(('Data-1', 'Data-2', 'Data-3', 'Data-4', 'Data-5', 'Data-6' ))
plt.show()


# # Range of Max and Min values

# In[9]:


maxval = train.iloc[:,1:].max(axis=1)
minval = train.iloc[:,1:].min(axis=1)

plt.figure(figsize=(20,5))
plt.title('Maximum flux value of each stars')
plt.xlabel('Index number of stars')
plt.ylabel('Max flux values')
plt.plot(np.arange(len(maxval)),maxval)


# In[10]:


plt.figure(figsize=(20,5))
plt.title('Minimum flux value of each stars')
plt.xlabel('Index number of stars')
plt.ylabel('Min flux values')
plt.plot(np.arange(len(minval)),minval)


# In[12]:


def reset(train,test):
    train_X = train.drop('LABEL', axis=1)
    train_y = train['LABEL'].values
    test_X = test.drop('LABEL', axis=1)
    test_y = test['LABEL'].values
    return train_X,train_y,test_X,test_y

train_X,train_y,test_X,test_y = reset(train,test)

# Different filters to be applied
def std_scaler(df1,df2):
    std_scaler = StandardScaler()
    train_X = std_scaler.fit_transform(df1)
    test_X = std_scaler.fit_transform(df2)
    return train_X,test_X

def norm(df1,df2):
    train_X = normalize(df1)
    test_X = normalize(df2)
    return train_X,test_X

def gaussian(df1,df2):
    train_X = ndimage.filters.gaussian_filter(df1, sigma=10)
    test_X = ndimage.filters.gaussian_filter(df2, sigma=10)
    return train_X,test_X

def fourier(df1,df2):
    train_X = np.abs(np.fft.fft(df1, axis=1))
    test_X = np.abs(np.fft.fft(df2, axis=1))
    return train_X,test_X

#For oversampling
def smote(a,b):
    model = SMOTE()
    X,y = model.fit_sample(a, b)
    return X,y


# In[13]:


#Machine learning models-

def logistic(train_X,train_y,test_X,test_y):
    lgr = LogisticRegression(max_iter=1000)
    lgr.fit(train_X,train_y)
    prediction_lgr=lgr.predict(test_X)
    print("-------------------------------------------")
    print("Logistic Regression")
    print("")
    print(classification_report(test_y,prediction_lgr))
    fig = plt.figure(figsize=(22,7))
    ax = fig.add_subplot(1,3,1)
    plot_confusion_matrix(lgr,test_X,test_y,ax=ax)
    ax = fig.add_subplot(1,3,2)
    metrics.plot_roc_curve(lgr,test_X, test_y,ax=ax)
    plt.plot([0, 1], [0, 1], 'k--')
    ax = fig.add_subplot(1,3,3)
    metrics.plot_precision_recall_curve(lgr, test_X, test_y,ax=ax)
    f1=metrics.f1_score(test_y, prediction_lgr,pos_label=2)
    print("F1 score of minority class:",f1)
    plt.show()
    return f1

def decisionTree(train_X,train_y,test_X,test_y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_X, train_y)
    y_pred_clf = clf.predict(test_X)
    print("-------------------------------------------")
    print("DecisionTree Classifier")
    print("")
    print(classification_report(test_y,y_pred_clf))
    fig = plt.figure(figsize=(22,7))
    ax = fig.add_subplot(1,3,1)
    plot_confusion_matrix(clf,test_X,test_y,ax=ax)
    ax = fig.add_subplot(1,3,2)
    metrics.plot_roc_curve(clf,test_X, test_y,ax=ax)
    plt.plot([0, 1], [0, 1], 'k--')
    ax = fig.add_subplot(1,3,3)
    metrics.plot_precision_recall_curve(clf, test_X, test_y,ax=ax)
    f1=metrics.f1_score(test_y, y_pred_clf,pos_label=2)
    print("F1 score of minority class:",f1)
    plt.show()
    return f1


def naiveBayes(train_X,train_y,test_X,test_y):
    gnb = GaussianNB()
    gnb.fit(train_X, train_y)
    y_pred=gnb.predict(test_X)
    print("-------------------------------------------")
    print("Gaussian NaiveBayes Classifier")
    print("")
    print(classification_report(test_y,y_pred))
    fig = plt.figure(figsize=(22,7))
    ax = fig.add_subplot(1,3,1)
    plot_confusion_matrix(gnb,test_X,test_y,ax=ax)
    ax = fig.add_subplot(1,3,2)
    metrics.plot_roc_curve(gnb,test_X, test_y,ax=ax)
    plt.plot([0, 1], [0, 1], 'k--')
    ax = fig.add_subplot(1,3,3)
    metrics.plot_precision_recall_curve(gnb, test_X, test_y,ax=ax)
    f1 = metrics.f1_score(test_y, y_pred,pos_label=2)
    print("F1 score of minority class:",f1)
    plt.show()
    return f1

def randomForest(train_X,train_y,test_X,test_y):
    rnd = RandomForestClassifier()
    rnd.fit(train_X, train_y)
    y_pred_rnd = rnd.predict(test_X)
    print("-------------------------------------------")
    print("Random Forest Classifier")
    print("")
    print(classification_report(test_y,y_pred_rnd))
    fig = plt.figure(figsize=(22,7))
    ax = fig.add_subplot(1,3,1)
    plot_confusion_matrix(rnd,test_X,test_y,ax=ax)
    ax = fig.add_subplot(1,3,2)
    metrics.plot_roc_curve(rnd,test_X,test_y,ax=ax)
    plt.plot([0, 1], [0, 1], 'k--') 
    ax = fig.add_subplot(1,3,3)
    metrics.plot_precision_recall_curve(rnd, test_X, test_y,ax=ax)
    plt.show()
    f1 = metrics.f1_score(test_y, y_pred_rnd,pos_label=2)
    print("F1 score of minority class:",f1)
    return f1

def knn(train_X,train_y,test_X,test_y):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_X, train_y)
    y_pred_neigh = neigh.predict(test_X)
    print("-------------------------------------------")
    print("k-Nearest Neighbour Classifier")
    print("")
    print(classification_report(test_y,y_pred_neigh))
    fig = plt.figure(figsize=(22,7))
    ax = fig.add_subplot(1,3,1)
    plot_confusion_matrix(neigh,test_X,test_y,ax=ax)
    ax = fig.add_subplot(1,3,2)
    metrics.plot_roc_curve(neigh,test_X, test_y,ax=ax)
    plt.plot([0, 1], [0, 1], 'k--')
    ax = fig.add_subplot(1,3,3)
    metrics.plot_precision_recall_curve(neigh, test_X, test_y,ax=ax)
    plt.show()
    f1 = metrics.f1_score(test_y, y_pred_neigh,pos_label=2)
    print("F1 score of minority class:",f1)
    return f1


# In[14]:


train_X,train_y,test_X,test_y = reset(train,test)
train_X,train_y = smote(train_X,train_y)
train_X, X, train_y, y = train_test_split(train_X, train_y, test_size=0.3)
test_X = np.concatenate((test_X, X), axis=0)
test_y = np.concatenate((test_y, y), axis=0)

print("Size of train dataset = ",train.shape)
print("Size of test dataset = ",test.shape) 
#train.shape, test.shape 
print("After oversampling")
print("Size of X-train dataset = ",train_X.shape)
print("Size of X-test dataset = ",test_X.shape)
print("Size of y-train dataset = ",train_y.shape)
print("Size of y-test dataset = ",test_y.shape)


# In[15]:


def robust(df1,df2):
    scaler = RobustScaler()
    train_X = scaler.fit_transform(df1)
    test_X = scaler.transform(df2)
    return train_X,test_X

train_X,train_y,test_X,test_y = reset(train,test)


train_X,test_X = std_scaler(train_X,test_X)
train_X,test_X = norm(train_X,test_X)
train_X,test_X = fourier(train_X,test_X)



train_X,train_y = smote(train_X,train_y)
train_X, X, train_y, y = train_test_split(train_X, train_y, test_size=0.3)
test_X = np.concatenate((test_X, X), axis=0)
test_y = np.concatenate((test_y, y), axis=0)

f1_smote =  []
f1_smote.append(logistic(train_X,train_y,test_X,test_y))
f1_smote.append(decisionTree(train_X,train_y,test_X,test_y))
f1_smote.append(linearSVC(train_X,train_y,test_X,test_y))
f1_smote.append(naiveBayes(train_X,train_y,test_X,test_y))
f1_smote.append(knn(train_X,train_y,test_X,test_y))
f1_smote.append(randomForest(train_X,train_y,test_X,test_y))
f1_smote


# # K-Means Clustering

# In[36]:


#K-means clustering
train_X,train_y,test_X,test_y = reset(train,test)

train_X,test_X = norm(train_X,test_X)
train_X,test_X = gaussian(train_X,test_X)
train_X,test_X = fourier(train_X,test_X)


train_X,train_y = smote(train_X,train_y)

pca = PCA(n_components=3)
pca.fit(test_X)
print(pca.explained_variance_ratio_)
pca_test_X = pca.transform(test_X)

km = KMeans(n_clusters=2)
y_predicted = km.fit_predict(test_X)
print(y_predicted[:40])
print(test_y[:40])


# In[37]:


y_predicted
#y_predicted = np.where(y_predicted == 1, 2, y_predicted)
y_predicted = np.where(y_predicted == 0, 2, y_predicted)


# In[38]:


print(classification_report(test_y,y_predicted))
print(confusion_matrix(test_y,y_predicted))


# In[39]:


pca_test_X = pd.DataFrame(pca_test_X)
a = pd.Series(test_y)
b = pd.Series(y_predicted)
pca_test_X = pd.concat([pca_test_X,a.rename('original'),b.rename('cluster')], axis=1)
pca_test_X.tail()


# In[41]:


from mpl_toolkits.mplot3d import Axes3D
x2 = pca_test_X[pca_test_X['original']==2]
x1 = pca_test_X[pca_test_X['original']==1]
y2 = pca_test_X[pca_test_X['cluster']==2]
y1 = pca_test_X[pca_test_X['cluster']==1]
fig = plt.figure(figsize=(15,7))

ax = fig.add_subplot(1,2,1, projection='3d')
ax.set_title('Original')
ax.scatter(x1.iloc[:,0],x1.iloc[:,1],x1.iloc[:,2],c='blue',s=5)
ax.scatter(x2.iloc[:,0],x2.iloc[:,1],x2.iloc[:,2],c='red',s=5)
ax = fig.add_subplot(1,2,2, projection='3d')
ax.set_title('K Means')
ax.scatter(y1.iloc[:,0],y1.iloc[:,1],y1.iloc[:,2],c='blue',s=5)
ax.scatter(y2.iloc[:,0],y2.iloc[:,1],y2.iloc[:,2],c='red',s=5)


# In[ ]:




