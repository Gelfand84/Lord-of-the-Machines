
# coding: utf-8

# Lord of the Machines: Data Science Hackathon

#%%

# Importing standard libraries
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interp
#from scipy.spatial.distance import cdist, pdist
#get_ipython().magic(u'matplotlib inline')

# Importing machine learning tools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc

# Importing text analysis tools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk.stem

#%%

## Reading files
raw_train=pd.read_csv('Data/train.csv', index_col='id')
raw_campaign=pd.read_csv('Data/campaign_data.csv')#, index_col='campaign_id')
raw_test=pd.read_csv('Data/test.csv', index_col='id')

#%%

## Checking file integrity

# Data header and types for train data
print raw_train.head()
print raw_train.info()
print raw_train.isnull().sum()

# Data header and types for campaign data
print raw_campaign.head()
print raw_campaign.info()
print raw_campaign.isnull().sum()

#%%

## Feature engineering

# Additional feature for campaign data
raw_campaign['no_of_internal_links']=raw_campaign.no_of_internal_links/raw_campaign.total_links
raw_campaign['no_of_images']=raw_campaign.no_of_images/raw_campaign.no_of_sections
raw_campaign['total_links']=raw_campaign.total_links/raw_campaign.no_of_sections

#%%

## Downsampling (arbitrary, not recommended)

#raw_train1=raw_train[raw_train.is_click==1]
#raw_train2=raw_train[raw_train.is_click==0].iloc[0:50000,:]
#raw_train=pd.concat([raw_train1, raw_train2])

#%%

# Stemming and Clustering for emails

# Import Stemmer and create class stemmer+vectorizer
english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

# List of texts    
emails=list(raw_campaign.email_body)
subjects=list(raw_campaign.subject)
documents=[x+' '+y for (x,y) in zip(subjects, emails)]

vectorizer = StemmedTfidfVectorizer(max_df=0.8, min_df=0.2, stop_words='english', norm='l2')
X = vectorizer.fit_transform(documents)

# Clustering
true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=99)
model.fit(X)

# Vector of predicted clusters
clusters=model.predict(vectorizer.transform(documents))

raw_campaign['cluster_id']=clusters

# Code below analizes clusters and elbow curve to determine true_k

#distortions=[]
#k_vector=range(true_k)=5
#
#for k in k_vect:
#    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
#    model.fit(X)
#    distortions.append(model.inertia_)
#    
#    print("Top terms per cluster:")
#    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
#    terms = vectorizer.get_feature_names()
#    for i in k_vector:
#        print("Cluster %d:" % i),
#        for ind in order_centroids[i, :10]:
#            print(' %s' % terms[ind]),
#        print
#     
#    print temp_list.append(model.inertia_)

#X=X.todense()
#
#K = range(1,52)
#KM = [KMeans(n_clusters=k, max_iter=100, n_init=1).fit(X) for k in K]
#centroids = [k.cluster_centers_ for k in KM]
#
#D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
#cIdx = [np.argmin(D,axis=1) for D in D_k]
#dist = [np.min(D,axis=1) for D in D_k]
#avgWithinSS = [sum(d)/X.shape[0] for d in dist]
#
## Total with-in sum of square
#wcss = [sum(d**2) for d in dist]
#tss = sum(pdist(X)**2)/X.shape[0]
#bss = tss-wcss
#
##kIdx = 10-1
#
#seg_threshold = 0.95 #Set this to your desired target
#
##The angle between three points
#def segments_gain(p1, v, p2):
#    vp1 = np.linalg.norm(p1 - v)
#    vp2 = np.linalg.norm(p2 - v)
#    p1p2 = np.linalg.norm(p1 - p2)
#    return np.arccos((vp1**2 + vp2**2 - p1p2**2) / (2 * vp1 * vp2)) / np.pi
#
##Normalize the data
#criterion = np.array(avgWithinSS)
#criterion = (criterion - criterion.min()) / (criterion.max() - criterion.min())
#
##Compute the angles
#seg_gains = np.array([0, ] + [segments_gain(*
#        [np.array([K[j], criterion[j]]) for j in range(i-1, i+2)]
#    ) for i in range(len(K) - 2)] + [np.nan, ])
#
##Get the first index satisfying the threshold
#kIdx = np.argmax(seg_gains > seg_threshold)
#
#
## elbow curve
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(K, avgWithinSS, 'b*-')
#ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
#markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
#plt.grid(True)
#plt.xlabel('Number of clusters')
#plt.ylabel('Average within-cluster sum of squares')
#plt.title('Elbow for KMeans clustering')
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(K, bss/tss*100, 'b*-')
#plt.grid(True)
#plt.xlabel('Number of clusters')
#plt.ylabel('Percentage of variance explained')
#plt.title('Elbow for KMeans clustering')    
   
################################

#%%

# Set datetime for train data
raw_train['send_date']=pd.to_datetime(raw_train.send_date, format='%d-%m-%Y %H:%M').copy()
raw_train['day_of_week']=raw_train.send_date.apply(lambda x: x.dayofweek)

# Set datetime for test data
raw_test['send_date']=pd.to_datetime(raw_test.send_date, format='%d-%m-%Y %H:%M').copy()
raw_test['day_of_week']=raw_test.send_date.apply(lambda x: x.dayofweek)

# Merging dataframes
raw_data=raw_train.reset_index().merge(raw_campaign, how='left').set_index(raw_train.index.names)
raw_data_test=raw_test.reset_index().merge(raw_campaign, how='left').set_index(raw_test.index.names)

#raw_data.drop(['no_of_internal_links', 'no_of_images', 'email_body', 'subject', 'email_url'], axis=1, inplace=True)
#raw_data_test.drop(['no_of_internal_links', 'no_of_images', 'email_body', 'subject', 'email_url'], axis=1, inplace=True)

raw_data.drop(['user_id', 'campaign_id', 'email_body', 'subject', 'email_url'], axis=1, inplace=True)
raw_data_test.drop(['user_id', 'campaign_id', 'email_body', 'subject', 'email_url'], axis=1, inplace=True)

cols=['day_of_week','total_links','cluster_id','no_of_internal_links','no_of_images', 'no_of_sections']+\
   ['communication_type']+['is_open', 'is_click']

raw_data=raw_data[cols]
raw_data_test=raw_data_test[cols[:-2]]

# Final datasets
data=raw_data.copy()
data_test=raw_data_test.copy()

#%%

## Exploratory analysis of campaign data

fig1=plt.figure(figsize=(16,8))
#fig1.suptitle('Campaign data')
plt.subplot(2,2,1)
sns.swarmplot(x='communication_type', y='total_links', data=raw_campaign)
plt.subplot(2,2,2)
sns.swarmplot(x='communication_type', y='no_of_internal_links', data=raw_campaign)
plt.subplot(2,2,3)
sns.swarmplot(x='communication_type', y='no_of_sections', data=raw_campaign)
plt.subplot(2,2,4)
sns.swarmplot(x='communication_type', y='no_of_images', data=raw_campaign)
plt.tight_layout()
plt.show()

#%%

## Exploratory analysis including train data

communication_type_list=['Newsletter','Conference','Upcoming Events','Others','Hackathon','Webinar','Corporate']
day_of_week_list=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# CTR vs communication_type
groupby_communication_type=raw_data.groupby('is_click')['communication_type'].value_counts()
ratio_vector=[100*groupby_communication_type.iloc[i+7]/(groupby_communication_type.iloc[i]+groupby_communication_type.iloc[i+7]) for i in np.arange(7)]

groupby_communication_type2=raw_data.groupby('is_open')['communication_type'].value_counts()
ratio_vector2=[100*groupby_communication_type2.iloc[i+7]/(groupby_communication_type2.iloc[i]+groupby_communication_type2.iloc[i+7]) for i in np.arange(7)]

plt.figure()
plt.bar(np.arange(7),ratio_vector2, label='is_open')
plt.bar(np.arange(7),ratio_vector, label='is_click')
plt.xticks(range(7), communication_type_list, rotation=40)
plt.ylabel('CTR [%]')
plt.grid(alpha=0.2, linestyle='--')
plt.legend()
plt.tight_layout()

# Groupby and factorplot (invalid after removing campaign_id from features)
#groupby_train=data.groupby(['campaign_id', 'is_open', 'is_click']).count()
#
#data_ratio=pd.DataFrame(index=np.arange(29,55))
#data_ratio['is_open']=np.nan
#data_ratio['is_click']=np.nan
#data_ratio['is_open_not_click']=np.nan
#
#for i in xrange(29,55):
#    denom=(groupby_train.loc[i,0,0]+groupby_train.loc[i,1,0]+groupby_train.loc[i,1,1])[0]
#    data_ratio['is_open'][i]=100*(1-groupby_train.loc[i,0,0][0]/denom)
#    data_ratio['is_click'][i]=100*groupby_train.loc[i,1,1][0]/denom
#    data_ratio['is_open_not_click'][i]=100*groupby_train.loc[i,1,0][0]/denom
#
#data_ratio.iloc[:,[2,1]].plot(kind='bar', stacked=True)
#plt.xlabel('campaign_id')
#plt.ylabel('Distribution per campaign [%]')
#plt.xticks(rotation=0)
#plt.grid(alpha=0.2, linestyle='--')
#plt.tight_layout()

# Noncategorial features
noncateg_idx=[1, 3, 4, 5]

fig=plt.figure(figsize=(6*2.13,4*2.13))
j=1
for i in noncateg_idx:
    plt.subplot(2,2,j)
    sns.distplot(data[data.is_click==1].iloc[:,i], norm_hist=True, kde=True, bins=40,  label='Clicked')
    sns.distplot(data[data.is_click==0].iloc[:,i], norm_hist=True, kde=True, bins=40,  label='Not clicked')
    j+=1
fig.legend(loc=7)
fig.tight_layout()

# Categorical features
N_is_click=data[data.is_click==1].shape[0]
N_not_click=data[data.is_click==0].shape[0]
N_is_open=data[data.is_open==1].shape[0]
N_not_open=data[data.is_open==0].shape[0]

ct_ratio_not_click=[np.nan]*7
ct_ratio_is_click=[np.nan]*7
ct_ratio_not_open=[np.nan]*7
ct_ratio_is_open=[np.nan]*7

dw_ratio_not_click=[np.nan]*7
dw_ratio_is_click=[np.nan]*7
dw_ratio_not_open=[np.nan]*7
dw_ratio_is_open=[np.nan]*7

for i in range(7):
    ct_ratio_is_click[i]=data[data.is_click==1]['communication_type'].value_counts(sort=False)[communication_type_list[i]]/N_is_click*100
    dw_ratio_is_click[i]=data[data.is_click==1]['day_of_week'].value_counts(sort=False)[i]/N_is_click*100
    ct_ratio_not_click[i]=data[data.is_click==0]['communication_type'].value_counts(sort=False)[communication_type_list[i]]/N_not_click*100
    dw_ratio_not_click[i]=data[data.is_click==0]['day_of_week'].value_counts(sort=False)[i]/N_not_click*100
    ct_ratio_is_open[i]=data[data.is_open==1]['communication_type'].value_counts(sort=False)[communication_type_list[i]]/N_is_open*100
    dw_ratio_is_open[i]=data[data.is_open==1]['day_of_week'].value_counts(sort=False)[i]/N_is_open*100
    ct_ratio_not_open[i]=data[data.is_open==0]['communication_type'].value_counts(sort=False)[communication_type_list[i]]/N_not_open*100
    dw_ratio_not_open[i]=data[data.is_open==0]['day_of_week'].value_counts(sort=False)[i]/N_not_open*100

barWidth=0.2
shifted_axis=[x+barWidth for x in range(7)]

plt.figure(figsize=(12,6))
plt.subplot(221)
plt.bar(range(7), ct_ratio_is_open, width=barWidth, label='is_open', alpha=0.8)
plt.bar(shifted_axis, ct_ratio_not_open, width=barWidth, label='not_open', alpha=0.8)
plt.xticks(range(7), communication_type_list, rotation=40)
plt.legend()
plt.grid(alpha=0.2, linestyle='--')
plt.ylabel('Opened/unopened emails [%]')

plt.subplot(222)
plt.bar(range(7), ct_ratio_is_click, width=barWidth, label='is_click', alpha=0.8)
plt.bar(shifted_axis, ct_ratio_not_click, width=barWidth, label='not_click', alpha=0.8)
plt.xticks(range(7), communication_type_list, rotation=40)
plt.legend()
plt.grid(alpha=0.2, linestyle='--')
plt.ylabel('Clicked/unclicked emails [%]')

plt.subplot(223)
plt.bar(range(7), dw_ratio_is_open, width=barWidth, label='is_open', alpha=0.8)
plt.bar(shifted_axis, dw_ratio_not_open, width=barWidth, label='not_open', alpha=0.8)
plt.xticks(range(7), day_of_week_list, rotation=40)
plt.legend()
plt.grid(alpha=0.2, linestyle='--')
plt.ylabel('Opened/unopened emails [%]')

plt.subplot(224)
plt.bar(range(7), dw_ratio_is_click, width=barWidth, label='is_click', alpha=0.8)
plt.bar(shifted_axis, dw_ratio_not_click, width=barWidth, label='not_click', alpha=0.8)
plt.xticks(range(7), day_of_week_list, rotation=40)
plt.legend()
plt.grid(alpha=0.2, linestyle='--')
plt.ylabel('Clicked/unclicked emails [%]')

plt.tight_layout()

#%%

## Prepare data for modeling

# Create dummy variables
dataset=pd.get_dummies(data, columns=['communication_type'], prefix='ct')
dataset=pd.get_dummies(dataset, columns=['day_of_week'], prefix='day')
dataset=pd.get_dummies(dataset, columns=['cluster_id'], prefix='cluster')

datatest=pd.get_dummies(data_test, columns=['communication_type'], prefix='cp')
datatest=pd.get_dummies(datatest, columns=['day_of_week'], prefix='day')
datatest=pd.get_dummies(datatest, columns=['cluster_id'], prefix='cluster')

# Sorting features (targets at the end)
num_cols=[0,1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,4,5]
dataset=dataset.iloc[:, num_cols]

# Adding features not present in dataset and datatest
#dataset=dataset[dataset.cp_Conference==0]
#dataset=dataset[dataset.cp_Others==0]
#dataset=dataset[dataset.cp_Webinar==0]
#dataset=dataset[dataset.day_5==0]
#dataset=dataset[dataset.day_6==0]
#dataset.drop(['cp_Conference', 'cp_Others', 'cp_Webinar', 'day_5', 'day_6'], axis=1, inplace=True)

datatest.insert(loc=4, column='cp_Conference', value=0)
datatest.insert(loc=8, column='cp_Others', value=0)
datatest.insert(loc=10, column='cp_Webinar', value=0)
datatest.insert(loc=16, column='day_5', value=0)
datatest.insert(loc=17, column='day_6', value=0)
datatest.insert(loc=22, column='cluster_4', value=0)

# Correlation matrix

corr_mat=dataset.corr(method='pearson')
plt.figure(figsize=(12,12))
sns.heatmap(corr_mat, vmax=1.0, square=True, annot=True)
plt.xticks(rotation=40)
plt.yticks(rotation=40)
plt.tight_layout()

# Separate atributes and is_open
# Here, we first predict is_open for test file and then train model again for is_click

X=dataset.iloc[:, :-2].values
Y=dataset.iloc[:, -2].values

#%%

## PCA (without target)
 
# StandardScaler
sc=StandardScaler()
X_pca=sc.fit_transform(X)

# PCA
pca=PCA().fit(X_pca)

# Explained variance plot
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,12,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.grid(True)

#%%

## Modeling

# Train/test split
seed=77
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.7, random_state=seed)
sc=StandardScaler().fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)

# Classification models (3 of them showed good results under time constraints)
models = []
models.append(('LR', LogisticRegression(class_weight='balanced', C=0.05)))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('QDA', QuadraticDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators=200, max_features=3, max_depth=3, class_weight='balanced', n_jobs=-1)))
#models.append(('GB', GradientBoostingClassifier()))
#models.append(('AB', AdaBoostClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM1', SVC(kernel="linear", C=0.025)))
#models.append(('SVM2', SVC(gamma=2, C=1)))

#%%

## Learning curves

def plot_learning_curve(estimator, title, X, Y, cv, train_sizes=np.linspace(.2, 1.0, 10)):
    #Generate a simple plot of the test and training learning curve
    plt.title(title)
    plt.ylim([0.6, 1.01])
    plt.xlabel('Training examples')
    plt.ylabel('AUC Score')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, Y, cv=cv, train_sizes=train_sizes, scoring='roc_auc',n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="CV score")
    plt.legend(loc=4)

# Plotting learning curves
fig=plt.figure(figsize=(6*2.13,4*2.13))
i=1
kfold = StratifiedKFold(n_splits=4)
for name, classifier in models:
    plt.subplot(1,3,i)
    plot_learning_curve(classifier, name+' learning curves', X_train, Y_train, cv=kfold) 
    i+=1
fig.tight_layout()

#%%

## Plotting ROC curves

kfold=StratifiedKFold(n_splits=8, random_state=99)
mean_fpr = np.linspace(0, 1, 100)
fig=plt.figure(figsize=(6*2.13,4*2.13))
i=0
j=1
for name, classifier in models[:9]:
    tprs = []
    aucs = []
    plt.subplot(1,3,j)
    for train, test in kfold.split(X_train, Y_train):
        probas=classifier.fit(X_train[train], Y_train[train]).predict_proba(X_train[test])
        fpr, tpr, thresholds = roc_curve(Y_train[test], probas[:,1], pos_label=1)
        tprs.append(interp(mean_fpr, fpr, tpr)) 
        tprs[-1][0]=0.0
        roc_auc=auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1)
        i+=1

    plt.plot([0,1], [0,1], linestyle='--')
    mean_tpr=np.mean(tprs, axis=0)
    mean_tpr[-1]=1.0
    mean_auc=auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean AUC (%0.2f)' %mean_auc, lw=2)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(title=name+' Classifier')
    j+=1
fig.tight_layout()

#%%

## Predicting is_open for test file

# Train/test split
seed=885
X_eval=datatest.copy().values

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.1, random_state=seed)
sc=StandardScaler().fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)
X_eval=sc.transform(X_eval)

# Model for is_open: 
model=RandomForestClassifier(n_estimators=500, max_features=3, max_depth=3, class_weight='balanced', n_jobs=-1)
#model=LogisticRegression(class_weight='balanced')
#model=GradientBoostingClassifier(verbose=1, max_features=1, n_estimators= 251, learning_rate= 0.1, max_depth=2)
#model=GaussianNB()
#model=KNeighborsClassifier(n_jobs=-1)

# Confusion matrix for is_open
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
conf_matrix=confusion_matrix(Y_test, Y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', linewidths=.5, cmap='winter', annot_kws={'size': 16}, alpha=0.8)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.title('Confusion matrix')
plt.show()
print 'Model score AUC: %0.4f' %roc_auc_score(Y_test, Y_pred)

#%%

# Writing is_open for test data
datatest['is_open']=model.predict(X_eval)
X_eval=datatest.copy().values

#%%

## Predicting is_click

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

seed=97
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=seed)
sc=StandardScaler().fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)
X_eval=sc.transform(X_eval)

# Modela for is_click

#model=LogisticRegression(class_weight='balanced', C=0.05)
#model=KNeighborsClassifier(n_jobs=-1)
#model=LinearDiscriminantAnalysis()
#model=QuadraticDiscriminantAnalysis()
#model=DecisionTreeClassifier()
model=RandomForestClassifier(n_estimators=778, max_features=4, max_depth=4, class_weight='balanced', n_jobs=-1)
#model=GradientBoostingClassifier(verbose=1, max_features=1, n_estimators= 251, learning_rate= 0.1, max_depth=2)
#model=AdaBoostClassifier(n_estimators=334, learning_rate=1.0)
#model=GaussianNB()
#model=SVC(class_weight='balanced')

# Confusion matrix for is_click
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
conf_matrix=confusion_matrix(Y_test, Y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', linewidths=.5, cmap='winter', annot_kws={'size': 16}, alpha=0.8)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.title('Confusion matrix')
plt.show()
print 'Model score AUC: %0.4f' %roc_auc_score(Y_test, Y_pred)

# Printing final result
Y_eval=model.predict(X_eval)
print Y_eval.mean()

df=pd.read_csv('Data/sample_submission.csv')
df['is_click']=Y_eval
df.to_csv('output6.csv', index=False)

#%% 

## Testing zone

# Calculating AUC scores for different methods and 20 folds
seed=44
results = []
names = []
scoring = 'roc_auc'
for name, model in models:
    print model
    kfold = StratifiedKFold(n_splits=20, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Boxplot comparing methods
fig, ax = plt.subplots(figsize=(6*2.13,4*2.13))
#plt.axhline(0.9685, alpha=0.5)
fig.suptitle('Algorithm Comparison (Score: UAC-ROC)')
plt.boxplot(results)
plt.grid(linestyle='dashed')
ax.set_xticklabels(names)
plt.ylabel('AUC')
plt.show()

#%%

## GridSearch for optimal parameters
#seed=19
#score='roc_auc'
#
## New train/test split and scaling
#X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.5, random_state=seed)
#sc=StandardScaler().fit(X_train)
#X_train=sc.transform(X_train)
#X_test=sc.transform(X_test)
#
## Parameters for GridSearch
#parameters_LR={'C': list(np.linspace(0.0001, 10, 200))}
#parameters_GB={'n_estimators': list(np.linspace(2,1000,5).astype(int)), 'learning_rate' : list(np.linspace(0.1,0.3,3)),\
#'max_depth': list(np.linspace(1,4,4).astype(int)), 'max_features': list(np.linspace(1,3,3).astype(int))}
#parameters_AB={'n_estimators': list(np.linspace(2,1000,10).astype(int)), 'learning_rate' : list(np.linspace(0.1,1,20))}
#parameters_RF={'n_estimators': list(np.linspace(100,1000,5).astype(int)), 'max_features': list(np.linspace(2,5,4).astype(int)), 
#'max_depth': list(np.linspace(2,5,4).astype(int))}
#parameters_SVC={'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}
#
## Determining optimal hyperparameters
#classifier_LR=LogisticRegression(random_state=seed, class_weight='balanced')
#gs_LR=GridSearchCV(classifier_LR, parameters_LR, n_jobs=-1, verbose=10, cv=10, scoring=score)
#gs_LR=gs_LR.fit(X_train,Y_train)
#gs_LR_bestParam = gs_LR.best_params_
#gs_LR_bestScore = gs_LR.best_score_
#print 'LR results:'
#print gs_LR_bestParam
#print gs_LR_bestScore

#classifier_GB=GradientBoostingClassifier(random_state=seed, class_weight='balanced')
#gs_GB=GridSearchCV(classifier_GB, parameters_GB, n_jobs=-1, verbose=10, cv=4, scoring=score)
#gs_GB=gs_GB.fit(X_train,Y_train)
#gs_GB_bestParam = gs_GB.best_params_
#gs_GB_bestScore = gs_GB.best_score_
#print 'GB results:'
#print gs_GB_bestParam
#print gs_GB_bestScore
#
#classifier_AB=AdaBoostClassifier(random_state=seed)
#gs_AB=GridSearchCV(classifier_AB, parameters_AB, n_jobs=-1, verbose=1, cv=4, scoring=score)
#gs_AB=gs_AB.fit(X_train,Y_train)
#gs_AB_bestParam = gs_AB.best_params_
#gs_AB_bestScore = gs_AB.best_score_
#print 'AB results:'
#print gs_AB_bestParam
#print gs_AB_bestScore

#classifier_RF=RandomForestClassifier(random_state=seed, class_weight='balanced')
#gs_RF=GridSearchCV(classifier_RF, parameters_RF, n_jobs=-1, verbose=10, cv=4, scoring=score)
#gs_RF=gs_RF.fit(X_train,Y_train)
#gs_RF_bestParam = gs_RF.best_params_
#gs_RF_bestScore = gs_RF.best_score_
#print 'RF results:'
#print gs_RF_bestParam
#print gs_RF_bestScore

#classifier_SVC=SVC(random_state=seed, class_weight='balanced')
#gs_SVC=GridSearchCV(classifier_SVC, parameters_SVC, n_jobs=-1, verbose=1, cv=4, scoring=score)
#gs_SVC=gs_SVC.fit(X_train,Y_train)
#gs_SVC_bestParam = gs_SVC.best_params_
#gs_SVC_bestScore = gs_SVC.best_score_
#print 'SVC results:'
#print gs_SVC_bestParam
#print gs_SVC_bestScore
