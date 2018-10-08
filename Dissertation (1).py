
# coding: utf-8

# In[4]:


get_ipython().system('pip install tpot')


# In[5]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras')


# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from scipy import interp

from tpot import TPOTClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix



from keras.models import Sequential
from keras.layers import Dense
import keras
import keras.callbacks as ct
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from scipy import interp
from sklearn.dummy import DummyClassifier


# In[2]:


psd_train_adr = "D:/Jack/UCL/Dissertation/data/turkey/psd_ds_train.csv"
psd_test_adr = "D:/Jack/UCL/Dissertation/data/turkey/psd_ds_test.csv"
psd_train = pd.read_csv(psd_train_adr, header=None)
psd_test = pd.read_csv(psd_test_adr, header=None)


# In[3]:


oxf_adr = "D:/Jack/UCL/Dissertation/data/oxford_dt.csv"
oxf_ds = pd.read_csv(oxf_adr)


# In[4]:


train_column_names = ['SubjectID','Jitter(local)','Jitter(local,absolute)','Jitter(rap)','Jitter(ppq5)','Jitter(ddp)',
                      'Shimmer(local)','Shimmer(local,dB)','Shimmer(apq3)','Shimmer(apq5)','Shimmer(apq11)','Shimmer(dda)',
                      'AC','NTH','HTN', 'MedianPitch', 'MeanPitch','StdPitch','MinimumPitch','MaximumPitch',
                      'Numberofpulses','Numberofperiods','Meanperiod','Stdofperiod',
                      'Fractionoflocallyunvoicedframes','Numberofvoicebreaks','Degreeofvoicebreaks','UPDRS','class']
psd_train.columns = train_column_names


# In[5]:


test_column_names = ['SubjectID','Jitter(local)','Jitter(local,absolute)','Jitter(rap)','Jitter(ppq5)','Jitter(ddp)',
                      'Shimmer(local)','Shimmer(local,dB)','Shimmer(apq3)','Shimmer(apq5)','Shimmer(apq11)','Shimmer(dda)',
                      'AC','NTH','HTN', 'MedianPitch', 'MeanPitch','StdPitch','MinimumPitch','MaximumPitch',
                      'Numberofpulses','Numberofperiods','Meanperiod','Stdofperiod',
                      'Fractionoflocallyunvoicedframes','Numberofvoicebreaks','Degreeofvoicebreaks','class']
psd_test.columns = test_column_names


# In[6]:


health_pat = 0
psd_ppl = 0

for i in range(len(psd_train)):
    if i != len(psd_train)-1:
        if psd_train['SubjectID'][i] != psd_train['SubjectID'][i+1]:
            if psd_train['class'][i] == 0:
                health_pat = health_pat + 1
            elif psd_train['class'][i] == 1:
                psd_ppl = psd_ppl + 1

    if i == len(psd_train)-1:
        if psd_train['class'][i] == 0:
            health_pat = health_pat + 1
        elif psd_train['class'][i] == 1:
            psd_ppl = psd_ppl + 1
            

print ("number of non psd patients: ", health_pat)
print ("number of psd patients: ",  psd_ppl)


# In[7]:


import seaborn as sns
sns.pairplot(psd_train,hue="class")


# In[8]:


psd_train


# In[9]:


oxf_ds.describe()


# In[10]:


psd_train.describe()


# In[11]:


psd_train_opt = psd_train.drop('UPDRS', 1)


# In[12]:


psd_train_opt


# In[13]:


psd_test


# In[14]:


psd_com = pd.concat([psd_train_opt, psd_test])


# In[15]:


psd_com.shape


# In[16]:


psd_com.describe()


# In[17]:


X, y = psd_com.iloc[:, 1:27].values, psd_com.iloc[:, -1].values


# In[18]:


X.shape


# In[19]:


psd_com.isnull().sum()


# In[20]:


oxf_ds.isnull().sum()


# In[21]:


#oxf_ds.isnull().sum()


# In[21]:


X_train_te, X_test, y_train_te, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[22]:


from imblearn.over_sampling import RandomOverSampler
OS = RandomOverSampler(random_state=15)

X_train, y_train = OS.fit_sample(X_train_te, y_train_te)


# In[23]:


stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# In[27]:


psd_com_std = stdsc.transform(X)


# In[28]:


plt.figure(figsize=(15,30))

ax = sns.boxplot(data=psd_com_std, orient="h", palette="Set2")


# In[25]:


sum(y==1)


# In[26]:


sum(y==0)


# In[27]:


sum(y_train==0)


# In[28]:


sum(y_train==1)


# In[29]:


sum(y_test==0)


# In[30]:


sum(y_test==1)


# In[14]:


#Sequential Backward Selection (SBS)
class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test =                 train_test_split(X, y, test_size=self.test_size, 
                                 random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


# In[196]:


svm = SVC(probability=True, verbose=False)

# selecting features
sbs = SBS(svm, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.5, 0.8])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# In[197]:


sbs.subsets_


# In[200]:


#what are those features that yielded such a good performance
k1=list(sbs.subsets_[8])
k1


# In[201]:


feature_labels = psd_train.columns[1:27]
feature_labels[k1]


# In[202]:


X_train_std_svm = X_train_std[:,k1]
X_test_std_svm = X_test_std[:,k1]


# In[31]:


pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


# In[32]:


plt.bar(np.arange(26), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(np.arange(26), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()


# In[33]:


pca = PCA(n_components=18)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


# In[34]:


print(abs( pca.components_ ))


# In[207]:


forest = RandomForestClassifier()

# selecting features
sbs = SBS(forest, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.5, 0.8])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# In[208]:


sbs.subsets_


# In[209]:


#what are those features that yielded such a good performance
k2=list(sbs.subsets_[4])
k2


# In[236]:


feature_labels = psd_train.columns[1:27]
feature_labels[k2]


# In[218]:


X_train_std_rf = X_train_std[:,k2]
X_test_std_rf = X_test_std[:,k2]


# In[219]:


pca = PCA()
X_train_pca_rf = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


# In[223]:


plt.bar(np.arange(26), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(np.arange(26), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()


# In[213]:


pca = PCA(n_components=11)
X_train_pca_rf = pca.fit_transform(X_train_std_rf)
pca.explained_variance_ratio_


# In[214]:


logistR = LogisticRegression()

# selecting features
sbs = SBS(logistR, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.5, 0.8])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# In[215]:


sbs.subsets_


# In[224]:


#what are those features that yielded such a good performance
k3=list(sbs.subsets_[10])
k3


# In[237]:


feature_labels = psd_train.columns[1:27]
feature_labels[k3]


# In[225]:


X_train_std_lr = X_train_std[:,k3]
X_test_std_lr = X_test_std[:,k3]


# In[226]:


pca = PCA()
X_train_pca_lr = pca.fit_transform(X_train_std_lr)
pca.explained_variance_ratio_


# In[228]:


plt.bar(np.arange(len(k3)), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(np.arange(len(k3)), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()


# In[229]:


pca = PCA(n_components=11)
X_train_pca_lr = pca.fit_transform(X_train_std_lr)
pca.explained_variance_ratio_


# In[230]:


knn = KNeighborsClassifier(n_neighbors=15)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.5, 0.8])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# In[231]:


sbs.subsets_


# In[232]:


#what are those features that yielded such a good performance
k4=list(sbs.subsets_[6])
k4


# In[238]:


feature_labels[k4]


# In[233]:


X_train_std_knn = X_train_std[:,k4]
X_test_std_knn = X_test_std[:,k4]


# In[234]:


pca = PCA()
X_train_pca_knn = pca.fit_transform(X_train_std_knn)
pca.explained_variance_ratio_


# In[235]:


plt.bar(np.arange(len(k4)), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(np.arange(len(k4)), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()


# In[239]:


pca = PCA(n_components=10)
X_train_pca_knn = pca.fit_transform(X_train_std_knn)
pca.explained_variance_ratio_


# In[23]:


NaivB = GaussianNB()

# selecting features
sbs = SBS(NaivB, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.5, 0.8])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# In[24]:


sbs.subsets_


# In[240]:


#what are those features that yielded such a good performance
k5=list(sbs.subsets_[18])
k5


# In[241]:


feature_labels[k5]


# In[242]:


X_train_std_nb = X_train_std[:,k5]
X_test_std_nb = X_test_std[:,k5]


# In[243]:


pca = PCA()
X_train_pca_nb = pca.fit_transform(X_train_std_nb)
pca.explained_variance_ratio_


# In[244]:


plt.bar(np.arange(len(k5)), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(np.arange(len(k5)), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()


# In[246]:


pca = PCA(n_components=7)
X_train_pca_nb = pca.fit_transform(X_train_std_nb)
pca.explained_variance_ratio_


# In[25]:


#tpot = TPOTClassifier(generations=5, population_size=50, cv=10,
#                                    random_state=42, verbosity=2)

#tpot.fit(X_train_std, y_train)


# In[26]:


#print(tpot.score(X_test_std, y_test))


# In[27]:


mlp = MLPClassifier(hidden_layer_sizes=(60,), max_iter=10, solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1)


# In[28]:


mlp.fit(X_train_std, y_train)
print("Training set score: %f" % mlp.score(X_train_std, y_train))
print("Test set score: %f" % mlp.score(X_test_std, y_test))


# In[29]:


mlp = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=200, alpha=1e-4,
                    activation='logistic', verbose=10, tol=1e-4, random_state=1)
mlp.fit(X_train_std, y_train)
print("\nTraining set score: %f" % mlp.score(X_train_std, y_train))
print("Test set score: %f" % mlp.score(X_test_std, y_test))

prediction = mlp.predict(X_test_std)

print(confusion_matrix(y_test,prediction))


# In[41]:


def create_model(activation='relu',neurons=20):
# create model
    model = Sequential()
    model.add(Dense(40, input_dim=26, activation='relu'))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[42]:


#Make Keras Classifier Pipeline
pipe_kc = Pipeline([('clf', KerasClassifier(build_fn=create_model, verbose=False))])

#Fit Pipeline to training Data
pipe_kc.fit(X_train_std, y_train)

num_folds = 5

scores = cross_val_score(estimator=pipe_kc, X=X_train_std, y=y_train, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters

param_grid = {'clf__neurons': [10,15,20],'clf__activation': ['sigmoid','relu','tanh'],
              'clf__epochs': [100],'clf__batch_size': [30,50]}

gs_kc = GridSearchCV(estimator=pipe_kc,
                  param_grid=param_grid,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_kc = gs_kc.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_kc.best_score_)
print('--> Best Parameters: \n',gs_kc.best_params_)


# In[ ]:


#FINALIZE MODEL
#Use best parameters
clf_kc = gs_kc.best_estimator_

#Get Final Scores
clf_kc.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_kc,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_kc.score(X_test_std,y_test))


# In[ ]:


#confusiong matrix KC

clf_kc.fit(X_train_std, y_train)
y_pred_kc = clf_kc.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_kc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_kc))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_kc))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_kc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_kc))


print(classification_report(y_test, y_pred_kc))


# In[36]:


pipe1 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=18)], ['SVC', SVC(probability=True, verbose=False)]])
pipe2 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=13)], ['RF', RandomForestClassifier()]])
pipe3 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=16)], ['LoR', LogisticRegression()]])
pipe4 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=20)], ['KNN', KNeighborsClassifier()]])               
pipe5 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=8)], ['GB', GaussianNB()]])    
pipe6 = Pipeline([['sc', StandardScaler()], ['MLP', MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000, alpha=1e-4,
                    activation='logistic', verbose=10, tol=1e-4, random_state=1)]])

estimators = []
estimators.append(('SVC', pipe1))
estimators.append(('RF', pipe2))
estimators.append(('LoR', pipe3))
estimators.append(('KNN', pipe4))
estimators.append(('GB', pipe5))
estimators.append(('MLP', pipe6))
clf_labels = ['Support Vector Machine', 'Random Forest', 'Logistic Regression', 'KNN','Gaussian NaiveBayes', 'MLP']


# In[38]:


# Hard Voting
ensemble = VotingClassifier(estimators, voting='hard')
all_clf = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, ensemble]
results = model_selection.cross_val_score(ensemble, X=X_train, y=y_train, cv=10)
results


# In[39]:


for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))


# In[40]:


# Soft Voting

ensemble = VotingClassifier(estimators, voting='soft')
all_clf = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, ensemble]
results_soft = model_selection.cross_val_score(ensemble, X=X_train, y=y_train, cv=10)
results_soft


# In[41]:



for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f) [%s]"  % (scores.mean(), scores.std(), label))


# In[34]:


#Make NB Pipeline
pipe_nb = Pipeline([('pca', PCA(n_components=18)),
                     ('clf', GaussianNB())])

#Fit Pipeline to training Data
pipe_nb.fit(X_train_std, y_train)

num_folds = 10

scores = cross_val_score(estimator=pipe_nb, X=X_train_std, y=y_train, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))


# In[35]:


#confusiong matrix NB

pipe_nb.fit(X_train_std, y_train)
y_pred_nb = pipe_nb.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_nb)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_nb))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_nb))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_nb))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_nb))


print(classification_report(y_test, y_pred_nb))


# In[36]:


#Dummy Classifier


# In[37]:


pipe_dc = Pipeline([('pca', PCA(n_components=18)),
                     ('clf', DummyClassifier())])

#Fit Pipeline to training Data
pipe_dc.fit(X_train_std, y_train)

num_folds = 10

scores = cross_val_score(estimator=pipe_dc, X=X_train_std, y=y_train, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))


# In[38]:


#confusiong matrix NB

pipe_dc.fit(X_train_std, y_train)
y_pred_dc = pipe_dc.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_dc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_dc))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_dc))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_dc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_dc))


print(classification_report(y_test, y_pred_dc))


# In[39]:


#Make Support Vector Classifier Pipeline
pipe_svc = Pipeline([('pca', PCA(n_components=18)),
                     ('clf', SVC(probability=True, verbose=False))])

#Fit Pipeline to training Data
pipe_svc.fit(X_train_std, y_train)

num_folds = 10

scores = cross_val_score(estimator=pipe_svc, X=X_train_std, y=y_train, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
param_range = [0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},
              {'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]

gs_svc = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_svc = gs_svc.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_svc.best_score_)
print('--> Best Parameters: \n',gs_svc.best_params_)


# In[40]:


#FINALIZE MODEL
#Use best parameters
clf_svc = gs_svc.best_estimator_

#Get Final Scores
clf_svc.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_svc,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_svc.score(X_test_std,y_test))


# In[41]:


#confusiong matrix SVM

clf_svc.fit(X_train_std, y_train)
y_pred_svc = clf_svc.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_svc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_svc))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_svc))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_svc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_svc))


print(classification_report(y_test, y_pred_svc))


# In[42]:


#Make Random Forest Pipeline

pipe_rf = Pipeline([('pca', PCA(n_components=18)),
                    ('clf', RandomForestClassifier())])

#Fit Pipeline to training Data
pipe_rf.fit(X_train_std, y_train)

num_folds = 10

scores = cross_val_score(estimator=pipe_rf, X=X_train_std, y=y_train, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
params = {'clf__criterion':['gini','entropy'],
          'clf__n_estimators':[10,15,20,25,30],
          'clf__min_samples_leaf':[1,2,3],
          'clf__min_samples_split':[3,4,5,6,7], 
          'clf__random_state':[1]}

gs_rf = GridSearchCV(estimator=pipe_rf,
                  param_grid=params,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=-1)

gs_rf = gs_rf.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_rf.best_score_)
print('--> Best Parameters: \n',gs_rf.best_params_)


# In[43]:


#FINALIZE MODEL
#Use best parameters
clf_rf = gs_rf.best_estimator_

#Get Final Scores
clf_rf.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_rf,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_rf.score(X_test_std,y_test))


# In[44]:


#confusiong matrix RF

clf_rf.fit(X_train_std, y_train)
y_pred_rf = clf_rf.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_rf)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_rf))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_rf))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_rf))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_rf))


print(classification_report(y_test, y_pred_rf))


# In[45]:


#Make Logistic Regression Classifier Pipeline
pipe_lr = Pipeline([('pca', PCA(n_components=18)),
                     ('clf', LogisticRegression())])

#Fit Pipeline to training Data
pipe_lr.fit(X_train_std, y_train)

num_folds = 10

scores = cross_val_score(estimator=pipe_lr, X=X_train_std, y=y_train, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = {'clf__C': param_range,'clf__penalty': ['l1', 'l2']}

gs_lr = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_lr = gs_lr.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_lr.best_score_)
print('--> Best Parameters: \n',gs_lr.best_params_)


# In[46]:


#FINALIZE MODEL
#Use best parameters
clf_lr = gs_lr.best_estimator_

#Get Final Scores
clf_lr.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_lr,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_lr.score(X_test_std,y_test))


# In[47]:


#confusiong matrix LogR

clf_lr.fit(X_train_std, y_train)
y_pred_lr = clf_lr.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_lr)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_lr))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_lr))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_lr))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_lr))


print(classification_report(y_test, y_pred_lr))


# In[48]:


#Make KNN Classifier Pipeline
pipe_knn = Pipeline([('pca', PCA(n_components=18)),
                     ('clf', KNeighborsClassifier())])

#Fit Pipeline to training Data
pipe_knn.fit(X_train_std, y_train)

num_folds = 10

scores = cross_val_score(estimator=pipe_knn, X=X_train_std, y=y_train, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

param_grid = {'clf__n_neighbors': param_range,'clf__weights': ['uniform', 'distance']}

gs_knn = GridSearchCV(estimator=pipe_knn,
                  param_grid=param_grid,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_knn = gs_knn.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_knn.best_score_)
print('--> Best Parameters: \n',gs_knn.best_params_)


# In[49]:


#FINALIZE MODEL
#Use best parameters
clf_knn = gs_knn.best_estimator_

#Get Final Scores
clf_knn.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_knn,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_knn.score(X_test_std,y_test))


# In[50]:


#confusiong matrix KNN

clf_knn.fit(X_train_std, y_train)
y_pred_knn = clf_knn.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_knn)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_knn))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_knn))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_knn))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_knn))


print(classification_report(y_test, y_pred_knn))


# In[51]:


#Make MLP Classifier Pipeline
pipe_mlp = Pipeline([('clf', MLPClassifier(max_iter=2000))])

#Fit Pipeline to training Data
pipe_mlp.fit(X_train_std, y_train)

num_folds = 10

scores = cross_val_score(estimator=pipe_mlp, X=X_train_std, y=y_train, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters

param_grid = {'clf__solver': ['lbfgs','sgd','adam'],'clf__hidden_layer_sizes': [(40,40),(50,50),(60,60),(70,70)],
              'clf__alpha': [0.001,0.0001],'clf__activation': ['logistic','relu']}

gs_mlp = GridSearchCV(estimator=pipe_mlp,
                  param_grid=param_grid,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_mlp = gs_mlp.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_mlp.best_score_)
print('--> Best Parameters: \n',gs_mlp.best_params_)


# In[52]:


#FINALIZE MODEL
#Use best parameters
clf_mlp = gs_mlp.best_estimator_

#Get Final Scores
clf_mlp.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_mlp,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_mlp.score(X_test_std,y_test))


# In[53]:


#confusiong matrix MLP

clf_mlp.fit(X_train_std, y_train)
y_pred_mlp = clf_mlp.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_mlp)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_mlp))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_mlp))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_mlp))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_mlp))


print(classification_report(y_test, y_pred_mlp))


# In[54]:


tpot = TPOTClassifier(generations=5, population_size=20, cv=10,
                                    random_state=42, verbosity=2)


# In[55]:


#confusiong matrix TPOT

tpot.fit(X_train_std, y_train)
y_pred_tpot = tpot.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_tpot)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_tpot))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_tpot))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_tpot))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_tpot))


print(classification_report(y_test, y_pred_tpot))


# In[56]:


def create_model(activation='relu',neurons=20):
# create model
    model = Sequential()
    model.add(Dense(40, input_dim=26, activation='relu'))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[57]:


#Make Keras Classifier Pipeline
pipe_kc = Pipeline([('clf', KerasClassifier(build_fn=create_model, verbose=False))])

#Fit Pipeline to training Data
pipe_kc.fit(X_train_std, y_train)

num_folds = 5

scores = cross_val_score(estimator=pipe_kc, X=X_train_std, y=y_train, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters

param_grid = {'clf__neurons': [10,15,20],'clf__activation': ['sigmoid','relu','tanh'],
              'clf__epochs': [100],'clf__batch_size': [30,50]}

gs_kc = GridSearchCV(estimator=pipe_kc,
                  param_grid=param_grid,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_kc = gs_kc.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_kc.best_score_)
print('--> Best Parameters: \n',gs_kc.best_params_)


# In[58]:


#FINALIZE MODEL
#Use best parameters
clf_kc = gs_kc.best_estimator_

#Get Final Scores
clf_kc.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_kc,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_kc.score(X_test_std,y_test))


# In[59]:


#confusiong matrix KC

clf_kc.fit(X_train_std, y_train)
y_pred_kc = clf_kc.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_kc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_kc))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_kc))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_kc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_kc))


print(classification_report(y_test, y_pred_kc))


# In[60]:


#ROC
fig = plt.figure(figsize=(8, 6))
all_tpr = []


probas = clf_svc.predict_proba(X_test_std)
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
current_auc = str('%.2f' %roc_auc)

probas_rf = clf_rf.predict_proba(X_test_std)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_true=y_test, y_score=probas_rf[:, 1], pos_label=1)
roc_auc = auc(fpr_rf, tpr_rf)
current_auc = str('%.2f' %roc_auc)

probas_lr = clf_lr.predict_proba(X_test_std)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_true=y_test, y_score=probas_lr[:, 1], pos_label=1)
roc_auc = auc(fpr_lr, tpr_lr)
current_auc = str('%.2f' %roc_auc)

probas_knn = clf_knn.predict_proba(X_test_std)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_true=y_test, y_score=probas_knn[:, 1], pos_label=1)
roc_auc = auc(fpr_knn, tpr_knn)
current_auc = str('%.2f' %roc_auc)

probas_mlp = clf_mlp.predict_proba(X_test_std)
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_true=y_test, y_score=probas_mlp[:, 1], pos_label=1)
roc_auc = auc(fpr_mlp, tpr_mlp)
current_auc = str('%.2f' %roc_auc)

probas_tpot = tpot.predict_proba(X_test_std)
fpr_tpot, tpr_tpot, thresholds_tpot = roc_curve(y_true=y_test, y_score=probas_tpot[:, 1], pos_label=1)
roc_auc = auc(fpr_tpot, tpr_tpot)
current_auc = str('%.2f' %roc_auc)

probas_kc = clf_kc.predict_proba(X_test_std)
fpr_kc, tpr_kc, thresholds_kc = roc_curve(y_true=y_test, y_score=probas_kc[:, 1], pos_label=1)
roc_auc = auc(fpr_kc, tpr_kc)
current_auc = str('%.2f' %roc_auc)

probas_nb = pipe_nb.predict_proba(X_test_std)
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_true=y_test, y_score=probas_nb[:, 1], pos_label=1)
roc_auc = auc(fpr_nb, tpr_nb)
current_auc = str('%.2f' %roc_auc)


plt.plot(fpr, 
         tpr, 
         lw=1,
         label='SVM')

plt.plot(fpr_rf, 
         tpr_rf, 
         lw=1,
         label='Random Forest')

plt.plot(fpr_lr, 
         tpr_lr, 
         lw=1,
         label='Logistic Regression')

plt.plot(fpr_knn, 
         tpr_knn, 
         lw=1,
         label='KNN')

plt.plot(fpr_mlp, 
         tpr_mlp, 
         lw=1,
         label='MLP')

plt.plot(fpr_tpot, 
         tpr_tpot, 
         lw=1,
         label='TPOT')

plt.plot(fpr_kc, 
         tpr_kc, 
         lw=1,
         label='KC')

plt.plot(fpr_nb, 
         tpr_nb, 
         lw=1,
         label='NB')

plt.plot([0, 1], 
         [0, 1], 
         linestyle='--', 
         color=(0.6, 0.6, 0.6), 
         label='random guessing')

plt.plot([0, 0, 1], 
         [0, 1, 1], 
         lw=2, 
         linestyle=':', 
         color='black', 
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[61]:


#F1


# In[68]:


num_folds = 10

#Tune Hyperparameters
param_range = [0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},
              {'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]

gs_svc_f1 = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='f1', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_svc_f1 = gs_svc_f1.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_svc_f1.best_score_)
print('--> Best Parameters: \n',gs_svc_f1.best_params_)


# In[69]:


#FINALIZE MODEL
#Use best parameters
clf_svc_f1 = gs_svc_f1.best_estimator_

#Get Final Scores
clf_svc_f1.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_svc_f1,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='f1',
                         n_jobs=1)
print('CV F1 scores: %s' % scores)
print('--> Final Model Training F1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final F1 on Test set: %.5f' % clf_svc_f1.score(X_test_std,y_test))


# In[70]:


#confusiong matrix SVM

clf_svc_f1.fit(X_train_std, y_train)
y_pred_svc = clf_svc_f1.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_svc)
print(confmat)


# In[71]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[72]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_svc))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_svc))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_svc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_svc))


# In[73]:


print(classification_report(y_test, y_pred_svc))


# In[74]:


# Random Forest Pipeline


#Tune Hyperparameters
params = {'clf__criterion':['gini','entropy'],
          'clf__n_estimators':[10,15,20,25,30],
          'clf__min_samples_leaf':[1,2,3],
          'clf__min_samples_split':[3,4,5,6,7], 
          'clf__random_state':[1]}

gs_rf_f1 = GridSearchCV(estimator=pipe_rf,
                  param_grid=params,
                  scoring='f1', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=-1)

gs_rf_f1 = gs_rf_f1.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_rf_f1.best_score_)
print('--> Best Parameters: \n',gs_rf_f1.best_params_)


# In[75]:


#FINALIZE MODEL
#Use best parameters
clf_rf_f1 = gs_rf_f1.best_estimator_

#Get Final Scores
clf_rf_f1.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_rf_f1,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='f1',
                         n_jobs=1)
print('CV f1 scores: %s' % scores)
print('--> Final Model Training f1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final f1 on Test set: %.5f' % clf_rf_f1.score(X_test_std,y_test))


# In[76]:


#confusiong matrix RF

clf_rf_f1.fit(X_train_std, y_train)
y_pred_rf = clf_rf_f1.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_rf)
print(confmat)


# In[77]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[78]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_rf))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_rf))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_rf))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_rf))


# In[79]:


print(classification_report(y_test, y_pred_rf))


# In[80]:


#LR Tune Hyperparameters
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = {'clf__C': param_range,'clf__penalty': ['l1', 'l2']}

gs_lr_f1 = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid,
                  scoring='f1', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_lr_f1 = gs_lr_f1.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_lr_f1.best_score_)
print('--> Best Parameters: \n',gs_lr_f1.best_params_)


# In[81]:


#FINALIZE MODEL
#Use best parameters
clf_lr_f1 = gs_lr_f1.best_estimator_

#Get Final Scores
clf_lr_f1.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_lr_f1,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='f1',
                         n_jobs=1)
print('CV f1 scores: %s' % scores)
print('--> Final Model Training f1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final f1 on Test set: %.5f' % clf_lr_f1.score(X_test_std,y_test))


# In[82]:


#confusiong matrix LogR

clf_lr_f1.fit(X_train_std, y_train)
y_pred_lr = clf_lr_f1.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_lr)
print(confmat)


# In[83]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[84]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_lr))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_lr))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_lr))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_lr))


# In[85]:


print(classification_report(y_test, y_pred_lr))


# In[86]:


#KNN Tune Hyperparameters
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

param_grid = {'clf__n_neighbors': param_range,'clf__weights': ['uniform', 'distance']}

gs_knn_f1 = GridSearchCV(estimator=pipe_knn,
                  param_grid=param_grid,
                  scoring='f1', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_knn_f1 = gs_knn_f1.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_knn_f1.best_score_)
print('--> Best Parameters: \n',gs_knn_f1.best_params_)


# In[87]:


#FINALIZE MODEL
#Use best parameters
clf_knn_f1 = gs_knn_f1.best_estimator_

#Get Final Scores
clf_knn_f1.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_knn_f1,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='f1',
                         n_jobs=1)
print('CV f1 scores: %s' % scores)
print('--> Final Model Training f1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final f1 on Test set: %.5f' % clf_knn_f1.score(X_test_std,y_test))


# In[88]:


#confusiong matrix KNN

clf_knn_f1.fit(X_train_std, y_train)
y_pred_knn = clf_knn_f1.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_knn)
print(confmat)


# In[89]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[90]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_knn))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_knn))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_knn))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_knn))


# In[91]:


print(classification_report(y_test, y_pred_knn))


# In[92]:


#Tune Hyperparameters MLP

param_grid = {'clf__solver': ['lbfgs','sgd','adam'],'clf__hidden_layer_sizes': [(40,40),(50,50),(60,60),(70,70)],
              'clf__alpha': [0.001,0.0001],'clf__activation': ['logistic','relu']}

gs_mlp_f1 = GridSearchCV(estimator=pipe_mlp,
                  param_grid=param_grid,
                  scoring='f1', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_mlp_f1 = gs_mlp_f1.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_mlp_f1.best_score_)
print('--> Best Parameters: \n',gs_mlp_f1.best_params_)


# In[93]:


#FINALIZE MODEL
#Use best parameters
clf_mlp_f1 = gs_mlp_f1.best_estimator_

#Get Final Scores
clf_mlp_f1.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_mlp_f1,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='f1',
                         n_jobs=1)
print('CV f1 scores: %s' % scores)
print('--> Final Model Training f1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final f1 on Test set: %.5f' % clf_mlp_f1.score(X_test_std,y_test))


# In[94]:


#confusiong matrix MLP

clf_mlp_f1.fit(X_train_std, y_train)
y_pred_mlp = clf_mlp_f1.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_mlp)
print(confmat)


# In[95]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[96]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_mlp))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_mlp))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_mlp))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_mlp))


# In[97]:


print(classification_report(y_test, y_pred_mlp))


# In[98]:


tpot = TPOTClassifier(generations=5, population_size=20, cv=10, scoring='f1',
                                    random_state=42, verbosity=2)


# In[99]:


#confusiong matrix TPOT

tpot.fit(X_train_std, y_train)
y_pred_tpot = tpot.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_tpot)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_tpot))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_tpot))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_tpot))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_tpot))


print(classification_report(y_test, y_pred_tpot))


# In[88]:


from sklearn import metrics
from keras import backend as K
import tensorflow as tf

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def auc_ro(y_true, y_pred):
    auc_ro = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc_ro


#https://datascience.stackexchange.com/questions/35775/how-to-find-auc-metric-value-for-keras-model
#https://github.com/keras-team/keras/issues/5400


# In[110]:


def create_model(activation='relu',neurons=20):
# create model
    model = Sequential()
    model.add(Dense(40, input_dim=26, activation='relu'))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    return model


# In[113]:


#Make Keras Classifier Pipeline
pipe_kc = Pipeline([('clf', KerasClassifier(build_fn=create_model, verbose=False))])

#Fit Pipeline to training Data
pipe_kc.fit(X_train_std, y_train)

num_folds = 5

#Tune Hyperparameters

param_grid = {'clf__neurons': [10,15,20],'clf__activation': ['sigmoid','relu','tanh'],
              'clf__epochs': [100],'clf__batch_size': [30,50]}

gs_kc = GridSearchCV(estimator=pipe_kc,
                  param_grid=param_grid,
                  scoring='f1', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_kc = gs_kc.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_kc.best_score_)
print('--> Best Parameters: \n',gs_kc.best_params_)


# In[116]:


#FINALIZE MODEL
#Use best parameters
clf_kc = gs_kc.best_estimator_


# In[117]:


#confusiong matrix KC

clf_kc.fit(X_train_std, y_train)
y_pred_kc = clf_kc.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_kc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_kc))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_kc))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_kc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_kc))


print(classification_report(y_test, y_pred_kc))


# In[121]:


#ROC
fig = plt.figure(figsize=(8, 6))
all_tpr = []


probas = clf_svc_f1.predict_proba(X_test_std)
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
current_auc = str('%.2f' %roc_auc)

probas_rf = clf_rf_f1.predict_proba(X_test_std)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_true=y_test, y_score=probas_rf[:, 1], pos_label=1)
roc_auc = auc(fpr_rf, tpr_rf)
current_auc = str('%.2f' %roc_auc)

probas_lr = clf_lr_f1.predict_proba(X_test_std)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_true=y_test, y_score=probas_lr[:, 1], pos_label=1)
roc_auc = auc(fpr_lr, tpr_lr)
current_auc = str('%.2f' %roc_auc)

probas_knn = clf_knn_f1.predict_proba(X_test_std)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_true=y_test, y_score=probas_knn[:, 1], pos_label=1)
roc_auc = auc(fpr_knn, tpr_knn)
current_auc = str('%.2f' %roc_auc)

probas_mlp = clf_mlp_f1.predict_proba(X_test_std)
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_true=y_test, y_score=probas_mlp[:, 1], pos_label=1)
roc_auc = auc(fpr_mlp, tpr_mlp)
current_auc = str('%.2f' %roc_auc)

probas_tpot = tpot.predict_proba(X_test_std)
fpr_tpot, tpr_tpot, thresholds_tpot = roc_curve(y_true=y_test, y_score=probas_tpot[:, 1], pos_label=1)
roc_auc = auc(fpr_tpot, tpr_tpot)
current_auc = str('%.2f' %roc_auc)

probas_kc = clf_kc.predict_proba(X_test_std)
fpr_kc, tpr_kc, thresholds_kc = roc_curve(y_true=y_test, y_score=probas_kc[:, 1], pos_label=1)
roc_auc = auc(fpr_kc, tpr_kc)
current_auc = str('%.2f' %roc_auc)

probas_nb = pipe_nb.predict_proba(X_test_std)
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_true=y_test, y_score=probas_nb[:, 1], pos_label=1)
roc_auc = auc(fpr_nb, tpr_nb)
current_auc = str('%.2f' %roc_auc)


plt.plot(fpr, 
         tpr, 
         lw=1,
         label='SVM')

plt.plot(fpr_rf, 
         tpr_rf, 
         lw=1,
         label='Random Forest')

plt.plot(fpr_lr, 
         tpr_lr, 
         lw=1,
         label='Logistic Regression')

plt.plot(fpr_knn, 
         tpr_knn, 
         lw=1,
         label='KNN')

plt.plot(fpr_mlp, 
         tpr_mlp, 
         lw=1,
         label='MLP')

plt.plot(fpr_tpot, 
         tpr_tpot, 
         lw=1,
         label='TPOT')

plt.plot(fpr_kc, 
         tpr_kc, 
         lw=1,
         label='KC')

plt.plot(fpr_nb, 
         tpr_nb, 
         lw=1,
         label='NB')

plt.plot([0, 1], 
         [0, 1], 
         linestyle='--', 
         color=(0.6, 0.6, 0.6), 
         label='random guessing')

plt.plot([0, 0, 1], 
         [0, 1, 1], 
         lw=2, 
         linestyle=':', 
         color='black', 
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[94]:


#precision


# In[122]:


#SVM Tune Hyperparameters
num_folds = 10

param_range = [0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},
              {'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]

gs_svc_p = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='precision', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_svc_p = gs_svc_p.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_svc_p.best_score_)
print('--> Best Parameters: \n',gs_svc_p.best_params_)


# In[123]:


#FINALIZE MODEL
#Use best parameters
clf_svc_p = gs_svc_p.best_estimator_

#Get Final Scores
clf_svc_p.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_svc_p,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='precision',
                         n_jobs=1)
print('CV F1 scores: %s' % scores)
print('--> Final Model Training F1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final F1 on Test set: %.5f' % clf_svc_p.score(X_test_std,y_test))


# In[124]:


#confusiong matrix SVM

clf_svc_p.fit(X_train_std, y_train)
y_pred_svc = clf_svc_p.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_svc)
print(confmat)


# In[125]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[126]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_svc))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_svc))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_svc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_svc))


# In[127]:


print(classification_report(y_test, y_pred_svc))


# In[128]:


# Random Forest Pipeline


#Tune Hyperparameters
params = {'clf__criterion':['gini','entropy'],
          'clf__n_estimators':[10,15,20,25,30],
          'clf__min_samples_leaf':[1,2,3],
          'clf__min_samples_split':[3,4,5,6,7], 
          'clf__random_state':[1]}

gs_rf_p = GridSearchCV(estimator=pipe_rf,
                  param_grid=params,
                  scoring='precision', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=-1)

gs_rf_p = gs_rf_p.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_rf_p.best_score_)
print('--> Best Parameters: \n',gs_rf_p.best_params_)


# In[129]:


#FINALIZE MODEL
#Use best parameters
clf_rf_p = gs_rf_p.best_estimator_

#Get Final Scores
clf_rf_p.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_rf_p,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='precision',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_rf_p.score(X_test_std,y_test))


# In[130]:


#confusiong matrix RF

clf_rf_p.fit(X_train_std, y_train)
y_pred_rf = clf_rf_p.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_rf)
print(confmat)


# In[131]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[132]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_rf))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_rf))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_rf))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_rf))


# In[133]:


print(classification_report(y_test, y_pred_rf))


# In[134]:


#LR Tune Hyperparameters
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = {'clf__C': param_range,'clf__penalty': ['l1', 'l2']}

gs_lr_p = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid,
                  scoring='precision', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_lr_p = gs_lr_p.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_lr_p.best_score_)
print('--> Best Parameters: \n',gs_lr_p.best_params_)


# In[135]:


#FINALIZE MODEL
#Use best parameters
clf_lr_p = gs_lr_p.best_estimator_

#Get Final Scores
clf_lr_p.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_lr_p,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='precision',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_lr_p.score(X_test_std,y_test))


# In[136]:


#confusiong matrix LogR

clf_lr_p.fit(X_train_std, y_train)
y_pred_lr = clf_lr_p.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_lr)
print(confmat)


# In[137]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[138]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_lr))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_lr))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_lr))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_lr))


# In[139]:


print(classification_report(y_test, y_pred_lr))


# In[140]:


#KNN Tune Hyperparameters
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

param_grid = {'clf__n_neighbors': param_range,'clf__weights': ['uniform', 'distance']}

gs_knn_p = GridSearchCV(estimator=pipe_knn,
                  param_grid=param_grid,
                  scoring='precision', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_knn_p = gs_knn_p.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_knn_p.best_score_)
print('--> Best Parameters: \n',gs_knn_p.best_params_)


# In[141]:


#FINALIZE MODEL
#Use best parameters
clf_knn_p = gs_knn_p.best_estimator_

#Get Final Scores
clf_knn_p.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_knn_p,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='precision',
                         n_jobs=1)
print('CV f1 scores: %s' % scores)
print('--> Final Model Training f1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final f1 on Test set: %.5f' % clf_knn_p.score(X_test_std,y_test))


# In[142]:


#confusiong matrix KNN

clf_knn_p.fit(X_train_std, y_train)
y_pred_knn = clf_knn_p.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_knn)
print(confmat)


# In[143]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[144]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_knn))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_knn))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_knn))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_knn))


# In[145]:


print(classification_report(y_test, y_pred_knn))


# In[146]:


#Tune Hyperparameters MLP

param_grid = {'clf__solver': ['lbfgs','sgd','adam'],'clf__hidden_layer_sizes': [(40,40),(50,50),(60,60),(70,70)],
              'clf__alpha': [0.001,0.0001],'clf__activation': ['logistic','relu']}

gs_mlp_p = GridSearchCV(estimator=pipe_mlp,
                  param_grid=param_grid,
                  scoring='precision', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_mlp_p = gs_mlp_p.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_mlp_p.best_score_)
print('--> Best Parameters: \n',gs_mlp_p.best_params_)


# In[147]:


#FINALIZE MODEL
#Use best parameters
clf_mlp_p = gs_mlp_p.best_estimator_

#Get Final Scores
clf_mlp_p.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_mlp_p,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='precision',
                         n_jobs=1)
print('CV f1 scores: %s' % scores)
print('--> Final Model Training f1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final f1 on Test set: %.5f' % clf_mlp_p.score(X_test_std,y_test))


# In[148]:


#confusiong matrix MLP

clf_mlp_p.fit(X_train_std, y_train)
y_pred_mlp = clf_mlp_p.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_mlp)
print(confmat)


# In[149]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[150]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_mlp))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_mlp))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_mlp))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_mlp))


# In[151]:


print(classification_report(y_test, y_pred_mlp))


# In[152]:


tpot = TPOTClassifier(generations=5, population_size=20, cv=10, scoring='precision',
                                    random_state=42, verbosity=2)


# In[153]:


#confusiong matrix TPOT

tpot.fit(X_train_std, y_train)
y_pred_tpot = tpot.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_tpot)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_tpot))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_tpot))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_tpot))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_tpot))


print(classification_report(y_test, y_pred_tpot))


# In[154]:


def create_model(activation='relu',neurons=20):
# create model
    model = Sequential()
    model.add(Dense(40, input_dim=26, activation='relu'))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[precision])
    return model


# In[155]:


#Make Keras Classifier Pipeline
pipe_kc = Pipeline([('clf', KerasClassifier(build_fn=create_model, verbose=False))])

#Fit Pipeline to training Data
pipe_kc.fit(X_train_std, y_train)

num_folds = 5

#Tune Hyperparameters

param_grid = {'clf__neurons': [10,15,20],'clf__activation': ['sigmoid','relu','tanh'],
              'clf__epochs': [100],'clf__batch_size': [30,50]}

gs_kc = GridSearchCV(estimator=pipe_kc,
                  param_grid=param_grid,
                  scoring='precision', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_kc = gs_kc.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_kc.best_score_)
print('--> Best Parameters: \n',gs_kc.best_params_)


# In[156]:


#FINALIZE MODEL
#Use best parameters
clf_kc = gs_kc.best_estimator_


# In[157]:


#confusiong matrix KC

clf_kc.fit(X_train_std, y_train)
y_pred_kc = clf_kc.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_kc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_kc))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_kc))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_kc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_kc))


print(classification_report(y_test, y_pred_kc))


# In[158]:


#ROC
fig = plt.figure(figsize=(8, 6))
all_tpr = []


probas = clf_svc_p.predict_proba(X_test_std)
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
current_auc = str('%.2f' %roc_auc)

probas_rf = clf_rf_p.predict_proba(X_test_std)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_true=y_test, y_score=probas_rf[:, 1], pos_label=1)
roc_auc = auc(fpr_rf, tpr_rf)
current_auc = str('%.2f' %roc_auc)

probas_lr = clf_lr_p.predict_proba(X_test_std)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_true=y_test, y_score=probas_lr[:, 1], pos_label=1)
roc_auc = auc(fpr_lr, tpr_lr)
current_auc = str('%.2f' %roc_auc)

probas_knn = clf_knn_p.predict_proba(X_test_std)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_true=y_test, y_score=probas_knn[:, 1], pos_label=1)
roc_auc = auc(fpr_knn, tpr_knn)
current_auc = str('%.2f' %roc_auc)

probas_mlp = clf_mlp_p.predict_proba(X_test_std)
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_true=y_test, y_score=probas_mlp[:, 1], pos_label=1)
roc_auc = auc(fpr_mlp, tpr_mlp)
current_auc = str('%.2f' %roc_auc)


probas_tpot = tpot.predict_proba(X_test_std)
fpr_tpot, tpr_tpot, thresholds_tpot = roc_curve(y_true=y_test, y_score=probas_tpot[:, 1], pos_label=1)
roc_auc = auc(fpr_tpot, tpr_tpot)
current_auc = str('%.2f' %roc_auc)

probas_nb = pipe_nb.predict_proba(X_test_std)
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_true=y_test, y_score=probas_nb[:, 1], pos_label=1)
roc_auc = auc(fpr_nb, tpr_nb)
current_auc = str('%.2f' %roc_auc)

probas_kc = clf_kc.predict_proba(X_test_std)
fpr_kc, tpr_kc, thresholds_kc = roc_curve(y_true=y_test, y_score=probas_kc[:, 1], pos_label=1)
roc_auc = auc(fpr_kc, tpr_kc)
current_auc = str('%.2f' %roc_auc)

plt.plot(fpr, 
         tpr, 
         lw=1,
         label='SVM')

plt.plot(fpr_rf, 
         tpr_rf, 
         lw=1,
         label='Random Forest')

plt.plot(fpr_lr, 
         tpr_lr, 
         lw=1,
         label='Logistic Regression')

plt.plot(fpr_knn, 
         tpr_knn, 
         lw=1,
         label='KNN')

plt.plot(fpr_mlp, 
         tpr_mlp, 
         lw=1,
         label='MLP')

plt.plot(fpr_tpot, 
         tpr_tpot, 
         lw=1,
         label='TPOT')

plt.plot(fpr_kc, 
         tpr_kc, 
         lw=1,
         label='KC')

plt.plot(fpr_nb, 
         tpr_nb, 
         lw=1,
         label='NB')

plt.plot([0, 1], 
         [0, 1], 
         linestyle='--', 
         color=(0.6, 0.6, 0.6), 
         label='random guessing')

plt.plot([0, 0, 1], 
         [0, 1, 1], 
         lw=2, 
         linestyle=':', 
         color='black', 
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[128]:


#Recall


# In[161]:


#SVM Tune Hyperparameters
num_folds = 10

param_range = [0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},
              {'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]

gs_svc_re = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='recall', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_svc_re = gs_svc_re.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_svc_re.best_score_)
print('--> Best Parameters: \n',gs_svc_re.best_params_)


# In[162]:


#FINALIZE MODEL
#Use best parameters
clf_svc_re = gs_svc_re.best_estimator_

#Get Final Scores
clf_svc_re.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_svc_re,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='recall',
                         n_jobs=1)
print('CV F1 scores: %s' % scores)
print('--> Final Model Training F1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final F1 on Test set: %.5f' % clf_svc_re.score(X_test_std,y_test))


# In[163]:


#confusiong matrix SVM

clf_svc_re.fit(X_train_std, y_train)
y_pred_svc = clf_svc_re.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_svc)
print(confmat)


# In[164]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[165]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_svc))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_svc))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_svc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_svc))


# In[166]:


print(classification_report(y_test, y_pred_svc))


# In[167]:


# Random Forest Pipeline


#Tune Hyperparameters
params = {'clf__criterion':['gini','entropy'],
          'clf__n_estimators':[10,15,20,25,30],
          'clf__min_samples_leaf':[1,2,3],
          'clf__min_samples_split':[3,4,5,6,7], 
          'clf__random_state':[1]}

gs_rf_re = GridSearchCV(estimator=pipe_rf,
                  param_grid=params,
                  scoring='recall', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=-1)

gs_rf_re = gs_rf_re.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_rf_re.best_score_)
print('--> Best Parameters: \n',gs_rf_re.best_params_)


# In[168]:


#FINALIZE MODEL
#Use best parameters
clf_rf_re = gs_rf_re.best_estimator_

#Get Final Scores
clf_rf_re.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_rf_re,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='recall',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_rf_re.score(X_test_std,y_test))


# In[169]:


#confusiong matrix RF

clf_rf_re.fit(X_train_std, y_train)
y_pred_rf = clf_rf_re.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_rf)
print(confmat)


# In[170]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[171]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_rf))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_rf))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_rf))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_rf))


# In[172]:


print(classification_report(y_test, y_pred_rf))


# In[173]:


#LR Tune Hyperparameters
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = {'clf__C': param_range,'clf__penalty': ['l1', 'l2']}

gs_lr_re = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid,
                  scoring='recall', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_lr_re = gs_lr_re.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_lr_re.best_score_)
print('--> Best Parameters: \n',gs_lr_re.best_params_)


# In[174]:


#FINALIZE MODEL
#Use best parameters
clf_lr_re = gs_lr_re.best_estimator_

#Get Final Scores
clf_lr_re.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_lr_re,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='recall',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_lr_re.score(X_test_std,y_test))


# In[175]:


#confusiong matrix LogR

clf_lr_re.fit(X_train_std, y_train)
y_pred_lr = clf_lr_re.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_lr)
print(confmat)


# In[176]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[177]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_lr))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_lr))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_lr))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_lr))


# In[178]:


print(classification_report(y_test, y_pred_lr))


# In[179]:


#KNN Tune Hyperparameters
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

param_grid = {'clf__n_neighbors': param_range,'clf__weights': ['uniform', 'distance']}

gs_knn_re = GridSearchCV(estimator=pipe_knn,
                  param_grid=param_grid,
                  scoring='recall', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_knn_re = gs_knn_re.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_knn_re.best_score_)
print('--> Best Parameters: \n',gs_knn_re.best_params_)


# In[180]:


#FINALIZE MODEL
#Use best parameters
clf_knn_re = gs_knn_re.best_estimator_

#Get Final Scores
clf_knn_re.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_knn_re,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='recall',
                         n_jobs=1)
print('CV f1 scores: %s' % scores)
print('--> Final Model Training f1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final f1 on Test set: %.5f' % clf_knn_re.score(X_test_std,y_test))


# In[181]:


#confusiong matrix KNN

clf_knn_re.fit(X_train_std, y_train)
y_pred_knn = clf_knn_re.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_knn)
print(confmat)


# In[182]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[183]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_knn))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_knn))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_knn))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_knn))


# In[184]:


print(classification_report(y_test, y_pred_knn))


# In[185]:


#Tune Hyperparameters MLP

param_grid = {'clf__solver': ['lbfgs','sgd','adam'],'clf__hidden_layer_sizes': [(40,40),(50,50),(60,60),(70,70)],
              'clf__alpha': [0.001,0.0001],'clf__activation': ['logistic','relu']}

gs_mlp_re = GridSearchCV(estimator=pipe_mlp,
                  param_grid=param_grid,
                  scoring='recall', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_mlp_re = gs_mlp_re.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_mlp_re.best_score_)
print('--> Best Parameters: \n',gs_mlp_re.best_params_)


# In[186]:


#FINALIZE MODEL
#Use best parameters
clf_mlp_re = gs_mlp_re.best_estimator_

#Get Final Scores
clf_mlp_re.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_mlp_re,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='recall',
                         n_jobs=1)
print('CV f1 scores: %s' % scores)
print('--> Final Model Training f1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final f1 on Test set: %.5f' % clf_mlp_re.score(X_test_std,y_test))


# In[187]:


#confusiong matrix MLP

clf_mlp_re.fit(X_train_std, y_train)
y_pred_mlp = clf_mlp_re.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_mlp)
print(confmat)


# In[188]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[189]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_mlp))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_mlp))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_mlp))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_mlp))


# In[190]:


print(classification_report(y_test, y_pred_mlp))


# In[191]:


tpot = TPOTClassifier(generations=5, population_size=20, cv=10, scoring='recall',
                                    random_state=42, verbosity=2)


# In[192]:


#confusiong matrix TPOT

tpot.fit(X_train_std, y_train)
y_pred_tpot = tpot.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_tpot)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_tpot))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_tpot))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_tpot))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_tpot))


print(classification_report(y_test, y_pred_tpot))


# In[193]:


def create_model(activation='relu',neurons=20):
# create model
    model = Sequential()
    model.add(Dense(40, input_dim=26, activation='relu'))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[recall])
    return model


# In[194]:


#Make Keras Classifier Pipeline
pipe_kc = Pipeline([('clf', KerasClassifier(build_fn=create_model, verbose=False))])

#Fit Pipeline to training Data
pipe_kc.fit(X_train_std, y_train)

num_folds = 5

#Tune Hyperparameters

param_grid = {'clf__neurons': [10,15,20],'clf__activation': ['sigmoid','relu','tanh'],
              'clf__epochs': [100],'clf__batch_size': [30,50]}

gs_kc = GridSearchCV(estimator=pipe_kc,
                  param_grid=param_grid,
                  scoring='recall', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_kc = gs_kc.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_kc.best_score_)
print('--> Best Parameters: \n',gs_kc.best_params_)


# In[195]:


#FINALIZE MODEL
#Use best parameters
clf_kc = gs_kc.best_estimator_


# In[196]:


#confusiong matrix KC

clf_kc.fit(X_train_std, y_train)
y_pred_kc = clf_kc.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_kc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_kc))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_kc))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_kc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_kc))


print(classification_report(y_test, y_pred_kc))


# In[197]:


#ROC
fig = plt.figure(figsize=(8, 6))
all_tpr = []


probas = clf_svc_re.predict_proba(X_test_std)
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
current_auc = str('%.2f' %roc_auc)

probas_rf = clf_rf_re.predict_proba(X_test_std)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_true=y_test, y_score=probas_rf[:, 1], pos_label=1)
roc_auc = auc(fpr_rf, tpr_rf)
current_auc = str('%.2f' %roc_auc)

probas_lr = clf_lr_re.predict_proba(X_test_std)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_true=y_test, y_score=probas_lr[:, 1], pos_label=1)
roc_auc = auc(fpr_lr, tpr_lr)
current_auc = str('%.2f' %roc_auc)

probas_knn = clf_knn_re.predict_proba(X_test_std)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_true=y_test, y_score=probas_knn[:, 1], pos_label=1)
roc_auc = auc(fpr_knn, tpr_knn)
current_auc = str('%.2f' %roc_auc)

probas_mlp = clf_mlp_re.predict_proba(X_test_std)
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_true=y_test, y_score=probas_mlp[:, 1], pos_label=1)
roc_auc = auc(fpr_mlp, tpr_mlp)
current_auc = str('%.2f' %roc_auc)

probas_tpot = tpot.predict_proba(X_test_std)
fpr_tpot, tpr_tpot, thresholds_tpot = roc_curve(y_true=y_test, y_score=probas_tpot[:, 1], pos_label=1)
roc_auc = auc(fpr_tpot, tpr_tpot)
current_auc = str('%.2f' %roc_auc)

probas_nb = pipe_nb.predict_proba(X_test_std)
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_true=y_test, y_score=probas_nb[:, 1], pos_label=1)
roc_auc = auc(fpr_nb, tpr_nb)
current_auc = str('%.2f' %roc_auc)

probas_kc = clf_kc.predict_proba(X_test_std)
fpr_kc, tpr_kc, thresholds_kc = roc_curve(y_true=y_test, y_score=probas_kc[:, 1], pos_label=1)
roc_auc = auc(fpr_kc, tpr_kc)
current_auc = str('%.2f' %roc_auc)

plt.plot(fpr, 
         tpr, 
         lw=1,
         label='SVM')

plt.plot(fpr_rf, 
         tpr_rf, 
         lw=1,
         label='Random Forest')

plt.plot(fpr_lr, 
         tpr_lr, 
         lw=1,
         label='Logistic Regression')

plt.plot(fpr_knn, 
         tpr_knn, 
         lw=1,
         label='KNN')

plt.plot(fpr_mlp, 
         tpr_mlp, 
         lw=1,
         label='MLP')

plt.plot(fpr_tpot, 
         tpr_tpot, 
         lw=1,
         label='TPOT')

plt.plot(fpr_kc, 
         tpr_kc, 
         lw=1,
         label='KC')

plt.plot(fpr_nb, 
         tpr_nb, 
         lw=1,
         label='NB')

plt.plot([0, 1], 
         [0, 1], 
         linestyle='--', 
         color=(0.6, 0.6, 0.6), 
         label='random guessing')

plt.plot([0, 0, 1], 
         [0, 1, 1], 
         lw=2, 
         linestyle=':', 
         color='black', 
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[162]:


#AUC


# In[198]:


#SVM Tune Hyperparameters
num_folds = 10

param_range = [0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},
              {'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]

gs_svc_roc = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='roc_auc', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_svc_roc = gs_svc_roc.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_svc_roc.best_score_)
print('--> Best Parameters: \n',gs_svc_roc.best_params_)


# In[199]:


#FINALIZE MODEL
#Use best parameters
clf_svc_roc = gs_svc_roc.best_estimator_

#Get Final Scores
clf_svc_roc.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_svc_roc,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='roc_auc',
                         n_jobs=1)
print('CV F1 scores: %s' % scores)
print('--> Final Model Training F1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final F1 on Test set: %.5f' % clf_svc_roc.score(X_test_std,y_test))


# In[200]:


#confusiong matrix SVM

clf_svc_roc.fit(X_train_std, y_train)
y_pred_svc = clf_svc_roc.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_svc)
print(confmat)


# In[201]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[202]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_svc))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_svc))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_svc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_svc))


# In[203]:


print(classification_report(y_test, y_pred_svc))


# In[204]:


# Random Forest Pipeline


#Tune Hyperparameters
params = {'clf__criterion':['gini','entropy'],
          'clf__n_estimators':[10,15,20,25,30],
          'clf__min_samples_leaf':[1,2,3],
          'clf__min_samples_split':[3,4,5,6,7], 
          'clf__random_state':[1]}

gs_rf_roc = GridSearchCV(estimator=pipe_rf,
                  param_grid=params,
                  scoring='roc_auc', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=-1)

gs_rf_roc = gs_rf_roc.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_rf_roc.best_score_)
print('--> Best Parameters: \n',gs_rf_roc.best_params_)


# In[205]:


#FINALIZE MODEL
#Use best parameters
clf_rf_roc = gs_rf_roc.best_estimator_

#Get Final Scores
clf_rf_roc.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_rf_roc,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='roc_auc',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_rf_roc.score(X_test_std,y_test))


# In[206]:


#confusiong matrix RF

clf_rf_roc.fit(X_train_std, y_train)
y_pred_rf = clf_rf_roc.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_rf)
print(confmat)


# In[207]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[208]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_rf))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_rf))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_rf))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_rf))


# In[209]:


print(classification_report(y_test, y_pred_rf))


# In[210]:


#LR Tune Hyperparameters
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = {'clf__C': param_range,'clf__penalty': ['l1', 'l2']}

gs_lr_roc = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid,
                  scoring='roc_auc', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_lr_roc = gs_lr_roc.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_lr_roc.best_score_)
print('--> Best Parameters: \n',gs_lr_roc.best_params_)


# In[211]:


#FINALIZE MODEL
#Use best parameters
clf_lr_roc = gs_lr_roc.best_estimator_

#Get Final Scores
clf_lr_roc.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_lr_roc,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='roc_auc',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_lr_roc.score(X_test_std,y_test))


# In[212]:


#confusiong matrix LogR

clf_lr_roc.fit(X_train_std, y_train)
y_pred_lr = clf_lr_roc.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_lr)
print(confmat)


# In[213]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[214]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_lr))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_lr))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_lr))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_lr))


# In[215]:


print(classification_report(y_test, y_pred_lr))


# In[216]:


#KNN Tune Hyperparameters
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

param_grid = {'clf__n_neighbors': param_range,'clf__weights': ['uniform', 'distance']}

gs_knn_roc = GridSearchCV(estimator=pipe_knn,
                  param_grid=param_grid,
                  scoring='roc_auc', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_knn_roc = gs_knn_roc.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_knn_roc.best_score_)
print('--> Best Parameters: \n',gs_knn_roc.best_params_)


# In[217]:


#FINALIZE MODEL
#Use best parameters
clf_knn_roc = gs_knn_roc.best_estimator_

#Get Final Scores
clf_knn_roc.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_knn_roc,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='roc_auc',
                         n_jobs=1)
print('CV f1 scores: %s' % scores)
print('--> Final Model Training f1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final f1 on Test set: %.5f' % clf_knn_roc.score(X_test_std,y_test))


# In[218]:


#confusiong matrix KNN

clf_knn_roc.fit(X_train_std, y_train)
y_pred_knn = clf_knn_roc.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_knn)
print(confmat)


# In[219]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[220]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_knn))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_knn))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_knn))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_knn))


# In[221]:


print(classification_report(y_test, y_pred_knn))


# In[222]:


#Tune Hyperparameters MLP

param_grid = {'clf__solver': ['lbfgs','sgd','adam'],'clf__hidden_layer_sizes': [(40,40),(50,50),(60,60),(70,70)],
              'clf__alpha': [0.001,0.0001],'clf__activation': ['logistic','relu']}

gs_mlp_roc = GridSearchCV(estimator=pipe_mlp,
                  param_grid=param_grid,
                  scoring='roc_auc', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_mlp_roc = gs_mlp_roc.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_mlp_roc.best_score_)
print('--> Best Parameters: \n',gs_mlp_roc.best_params_)


# In[223]:


#FINALIZE MODEL
#Use best parameters
clf_mlp_roc = gs_mlp_roc.best_estimator_

#Get Final Scores
clf_mlp_roc.fit(X_train_std, y_train)
scores = cross_val_score(estimator=clf_mlp_roc,
                         X=X_train_std,
                         y=y_train,
                         cv=num_folds,
                         scoring='roc_auc',
                         n_jobs=1)
print('CV f1 scores: %s' % scores)
print('--> Final Model Training f1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final f1 on Test set: %.5f' % clf_mlp_roc.score(X_test_std,y_test))


# In[224]:


#confusiong matrix MLP

clf_mlp_roc.fit(X_train_std, y_train)
y_pred_mlp = clf_mlp_roc.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_mlp)
print(confmat)


# In[225]:


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


# In[226]:


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_mlp))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_mlp))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_mlp))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_mlp))


# In[227]:


print(classification_report(y_test, y_pred_mlp))


# In[228]:


tpot = TPOTClassifier(generations=5, population_size=20, cv=10, scoring='roc_auc',
                                    random_state=42, verbosity=2)


# In[229]:


#confusiong matrix TPOT

tpot.fit(X_train_std, y_train)
y_pred_tpot = tpot.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_tpot)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_tpot))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_tpot))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_tpot))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_tpot))


print(classification_report(y_test, y_pred_tpot))


# In[230]:


def create_model(activation='relu',neurons=20):
# create model
    model = Sequential()
    model.add(Dense(40, input_dim=26, activation='relu'))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_ro])
    return model


# In[231]:


#Make Keras Classifier Pipeline
pipe_kc = Pipeline([('clf', KerasClassifier(build_fn=create_model, verbose=False))])

#Fit Pipeline to training Data
pipe_kc.fit(X_train_std, y_train)

num_folds = 5

#Tune Hyperparameters

param_grid = {'clf__neurons': [10,15,20],'clf__activation': ['sigmoid','relu','tanh'],
              'clf__epochs': [100],'clf__batch_size': [30,50]}

gs_kc = GridSearchCV(estimator=pipe_kc,
                  param_grid=param_grid,
                  scoring='roc_auc', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_kc = gs_kc.fit(X_train_std, y_train)
print('--> Tuned Parameters Best Score: ',gs_kc.best_score_)
print('--> Best Parameters: \n',gs_kc.best_params_)


# In[232]:


#FINALIZE MODEL
#Use best parameters
clf_kc = gs_kc.best_estimator_



# In[233]:


#confusiong matrix KC

clf_kc.fit(X_train_std, y_train)
y_pred_kc = clf_kc.predict(X_test_std)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_kc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred_kc))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred_kc))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred_kc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred_kc))


print(classification_report(y_test, y_pred_kc))


# In[234]:


#ROC
fig = plt.figure(figsize=(8, 6))
all_tpr = []


probas = clf_svc_roc.predict_proba(X_test_std)
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
current_auc = str('%.2f' %roc_auc)

probas_rf = clf_rf_roc.predict_proba(X_test_std)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_true=y_test, y_score=probas_rf[:, 1], pos_label=1)
roc_auc = auc(fpr_rf, tpr_rf)
current_auc = str('%.2f' %roc_auc)

probas_lr = clf_lr_roc.predict_proba(X_test_std)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_true=y_test, y_score=probas_lr[:, 1], pos_label=1)
roc_auc = auc(fpr_lr, tpr_lr)
current_auc = str('%.2f' %roc_auc)

probas_knn = clf_knn_roc.predict_proba(X_test_std)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_true=y_test, y_score=probas_knn[:, 1], pos_label=1)
roc_auc = auc(fpr_knn, tpr_knn)
current_auc = str('%.2f' %roc_auc)

probas_mlp = clf_mlp_roc.predict_proba(X_test_std)
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_true=y_test, y_score=probas_mlp[:, 1], pos_label=1)
roc_auc = auc(fpr_mlp, tpr_mlp)
current_auc = str('%.2f' %roc_auc)

probas_tpot = tpot.predict_proba(X_test_std)
fpr_tpot, tpr_tpot, thresholds_tpot = roc_curve(y_true=y_test, y_score=probas_tpot[:, 1], pos_label=1)
roc_auc = auc(fpr_tpot, tpr_tpot)
current_auc = str('%.2f' %roc_auc)

probas_nb = pipe_nb.predict_proba(X_test_std)
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_true=y_test, y_score=probas_nb[:, 1], pos_label=1)
roc_auc = auc(fpr_nb, tpr_nb)
current_auc = str('%.2f' %roc_auc)

probas_kc = clf_kc.predict_proba(X_test_std)
fpr_kc, tpr_kc, thresholds_kc = roc_curve(y_true=y_test, y_score=probas_kc[:, 1], pos_label=1)
roc_auc = auc(fpr_kc, tpr_kc)
current_auc = str('%.2f' %roc_auc)

plt.plot(fpr, 
         tpr, 
         lw=1,
         label='SVM')

plt.plot(fpr_rf, 
         tpr_rf, 
         lw=1,
         label='Random Forest')

plt.plot(fpr_lr, 
         tpr_lr, 
         lw=1,
         label='Logistic Regression')

plt.plot(fpr_knn, 
         tpr_knn, 
         lw=1,
         label='KNN')

plt.plot(fpr_mlp, 
         tpr_mlp, 
         lw=1,
         label='MLP')

plt.plot(fpr_tpot, 
         tpr_tpot, 
         lw=1,
         label='TPOT')

plt.plot(fpr_nb, 
         tpr_nb, 
         lw=1,
         label='NB')

plt.plot(fpr_kc, 
         tpr_kc, 
         lw=1,
         label='KC')

plt.plot([0, 1], 
         [0, 1], 
         linestyle='--', 
         color=(0.6, 0.6, 0.6), 
         label='random guessing')

plt.plot([0, 0, 1], 
         [0, 1, 1], 
         lw=2, 
         linestyle=':', 
         color='black', 
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[56]:


#psd_train_df


# In[52]:


#columns = ['UPDRS_o', 'UPDRS_u',
#           'UPDRS_1','UPDRS_2','UPDRS_3','UPDRS_4','UPDRS_5',
#           'UPDRS_6','UPDRS_7','UPDRS_8','UPDRS_9','UPDRS_10',
#           'UPDRS_ss1','UPDRS_ss2','UPDRS_ss3','UPDRS_ss4',
#           'UPDRS_w1','UPDRS_w2','UPDRS_w3','UPDRS_w4','UPDRS_w5',
#           'UPDRS_w6','UPDRS_w7','UPDRS_w8','UPDRS_w9',
#                        
#           'class_o', 'class_u',
#           'class_1','class_2','class_3','class_4','class_5',
#           'class_6','class_7','class_8','class_9','class_10',
#           'class_ss1','class_ss2','class_ss3','class_ss4',
#           'class_w1','class_w2','class_w3','class_w4','class_w5',
#           'class_w6','class_w7','class_w8','class_w9']
#psd_train_df.drop(columns, inplace=True, axis=1)


# In[57]:


#psd_train_df


# In[55]:


#psd_train_df.shape


# In[59]:


#psd_train_df.describe()


# In[67]:


#X_opt1, y_opt1 = psd_train_df.iloc[:, 1:677].values, psd_train_df.iloc[:, -1].values


# In[68]:


#X_train_opt1, X_test_opt1, y_train_opt1, y_test_opt1 = train_test_split(X_opt1, y_opt1, test_size=0.2, random_state=1)


# In[69]:


#stdsc = StandardScaler()
#X_train_opt1_std = stdsc.fit_transform(X_train_opt1)
#X_test_opt1_std = stdsc.transform(X_test_opt1)


# In[82]:


#svm = SVC(probability=True, verbose=False)

# selecting features
#sbs = SBS(svm, k_features=1)
#sbs.fit(X_train_opt1_std, y_train_opt1)

# plotting performance of feature subsets
#k_feat = [len(k) for k in sbs.subsets_]
#
#plt.plot(k_feat, sbs.scores_, marker='o')
#plt.ylim([0, 1])
#plt.ylabel('Accuracy')
#plt.xlabel('Number of features')
#plt.grid()
#plt.tight_layout()
#plt.show()


# In[58]:


#sbs.subsets_


# In[83]:


#forest = RandomForestClassifier()

# selecting features
#sbs = SBS(forest, k_features=1)
#sbs.fit(X_train_opt1_std, y_train_opt1)

# plotting performance of feature subsets
#k_feat = [len(k) for k in sbs.subsets_]

#plt.plot(k_feat, sbs.scores_, marker='o')
#plt.ylim([0.5, 1])
#plt.ylabel('Accuracy')
#plt.xlabel('Number of features')
#plt.grid()
#plt.tight_layout()
#plt.show()


# In[59]:


#sbs.subsets_


# In[88]:


#logistR = LogisticRegression()

# selecting features
#sbs = SBS(logistR, k_features=1)
#sbs.fit(X_train_opt1_std, y_train_opt1)

# plotting performance of feature subsets
#k_feat = [len(k) for k in sbs.subsets_]

#plt.plot(k_feat, sbs.scores_, marker='o')
#plt.ylim([0, 1])
#plt.ylabel('Accuracy')
#plt.xlabel('Number of features')
#plt.grid()
#plt.tight_layout()
#plt.show()


# In[60]:


#sbs.subsets_


# In[84]:


#knn = KNeighborsClassifier(n_neighbors=5)

# selecting features
#sbs = SBS(knn, k_features=1)
#sbs.fit(X_train_opt1_std, y_train_opt1)

# plotting performance of feature subsets
#k_feat = [len(k) for k in sbs.subsets_]

#plt.plot(k_feat, sbs.scores_, marker='o')
#plt.ylim([0, 1])
#plt.ylabel('Accuracy')
#plt.xlabel('Number of features')
#plt.grid()
#plt.tight_layout()
#plt.show()


# In[61]:


#sbs.subsets_


# In[85]:


#NaivB = GaussianNB()

# selecting features
#sbs = SBS(NaivB, k_features=1)
#sbs.fit(X_train_opt1_std, y_train_opt1)

# plotting performance of feature subsets
#k_feat = [len(k) for k in sbs.subsets_]

#plt.plot(k_feat, sbs.scores_, marker='o')
#plt.ylim([0, 1])
#plt.ylabel('Accuracy')
#plt.xlabel('Number of features')
#plt.grid()
#plt.tight_layout()
#plt.show()


# In[62]:


#sbs.subsets_


# In[86]:


#tpot = TPOTClassifier(generations=5, population_size=20, cv=10,
#                                    random_state=42, verbosity=2)

#tpot.fit(X_train_opt1_std, y_train_opt1)


# In[87]:


#mlp = MLPClassifier(hidden_layer_sizes=(300,300), max_iter=250, alpha=1e-4,
#                    activation='logistic', verbose=10, tol=1e-4, random_state=1)
#mlp.fit(X_train_opt1_std, y_train_opt1)
#print("\nTraining set score: %f" % mlp.score(X_train_opt1_std, y_train_opt1))
#print("Test set score: %f" % mlp.score(X_test_opt1_std, y_test_opt1))

#prediction = mlp.predict(X_test_opt1_std)

#print(confusion_matrix(y_test_opt1,prediction))


# In[127]:


def create_model():
# create model
    model = Sequential()
    model.add(Dense(500, input_dim=676, activation='relu'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[128]:


#we can increase the epochs
model = KerasClassifier(build_fn=create_model, epochs=90, batch_size=50, verbose=False)

#try this only at home as it takes a long computing time
model_cross_val = cross_val_score(model, X_train_opt1_std, y_train_opt1, cv=5).mean()

print('Model cross validation score %.5f'% (model_cross_val))


# In[129]:


model.fit(X_train_opt1_std, y_train_opt1)
y_train_pred = model.predict(X_train_opt1_std)
y_test_pred = model.predict(X_test_opt1_std)

model_train = accuracy_score(y_train_opt1, y_train_pred) 
model_test = accuracy_score(y_test_opt1, y_test_pred) 
print('Model train accuracies %.5f'% (model_train))
print('Model test accuracies %.5f'% (model_test))


# In[130]:


pipe1 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=20)], ['SVC', SVC(probability=True, verbose=False)]])
pipe2 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=50)], ['RF', RandomForestClassifier()]])
pipe3 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=100)], ['LoR', LogisticRegression()]])
pipe4 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=100)], ['KNN', KNeighborsClassifier()]])               
pipe5 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=8)], ['GB', GaussianNB()]])    
pipe6 = Pipeline([['sc', StandardScaler()], ['MLP', MLPClassifier(hidden_layer_sizes=(70,70), max_iter=250, alpha=1e-4,
                    activation='logistic', verbose=10, tol=1e-4, random_state=1)]])

estimators = []
estimators.append(('SVC', pipe1))
estimators.append(('RF', pipe2))
estimators.append(('LoR', pipe3))
estimators.append(('KNN', pipe4))
estimators.append(('GB', pipe5))
estimators.append(('MLP', pipe6))
clf_labels = ['Support Vector Machine', 'Random Forest', 'Logistic Regression', 'KNN','Gaussian NaiveBayes', 'MLP']


# In[132]:


# Hard Voting
ensemble = VotingClassifier(estimators, voting='hard')
all_clf = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, ensemble]
results = model_selection.cross_val_score(ensemble, X=X_train_opt1, y=y_train_opt1, cv=5)
results


# In[133]:


for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train_opt1, y=y_train_opt1, cv=5, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))


# In[134]:


# Soft Voting

ensemble = VotingClassifier(estimators, voting='soft')
all_clf = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, ensemble]
results = model_selection.cross_val_score(ensemble, X=X_train_opt1, y=y_train_opt1, cv=5)
results


# In[135]:


for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train_opt1, y=y_train_opt1, cv=5, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f) [%s]"  % (scores.mean(), scores.std(), label))


# In[145]:


#Make Support Vector Classifier Pipeline
pipe_svc = Pipeline([('pca', PCA(n_components=10)),
                     ('clf', SVC(probability=True, verbose=False))])

#Fit Pipeline to training Data
pipe_svc.fit(X_train_opt1_std, y_train_opt1)

num_folds = 5

scores = cross_val_score(estimator=pipe_svc, X=X_train_opt1_std, y=y_train_opt1, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},
              {'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]

gs_svc = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_svc = gs_svc.fit(X_train_opt1_std, y_train_opt1)
print('--> Tuned Parameters Best Score: ',gs_svc.best_score_)
print('--> Best Parameters: \n',gs_svc.best_params_)


# In[148]:


#FINALIZE MODEL
#Use best parameters
clf_svc = gs_svc.best_estimator_

#Get Final Scores
clf_svc.fit(X_train_opt1_std, y_train_opt1)
scores = cross_val_score(estimator=clf_svc,
                         X=X_train_opt1_std,
                         y=y_train_opt1,
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_svc.score(X_test_opt1_std,y_test_opt1))


# In[ ]:


#Oxford


# In[29]:


oxf_ds


# In[30]:


oxf_ds.describe()


# In[31]:


cols = list(oxf_ds.columns.values)
cols


# In[32]:


oxf_ds = oxf_ds[['name',
 'MDVP:Fo(Hz)',
 'MDVP:Fhi(Hz)',
 'MDVP:Flo(Hz)',
 'MDVP:Jitter(%)',
 'MDVP:Jitter(Abs)',
 'MDVP:RAP',
 'MDVP:PPQ',
 'Jitter:DDP',
 'MDVP:Shimmer',
 'MDVP:Shimmer(dB)',
 'Shimmer:APQ3',
 'Shimmer:APQ5',
 'MDVP:APQ',
 'Shimmer:DDA',
 'NHR',
 'HNR',
 'RPDE',
 'DFA',
 'spread1',
 'spread2',
 'D2',
 'PPE',
 'status']]


# In[33]:


oxf_ds


# In[34]:


X1, y1 = oxf_ds.iloc[:, 1:23].values, oxf_ds.iloc[:, -1].values
X_train_tes, X_test_ox, y_train_tes, y_test_ox = train_test_split(X1, y1, test_size=0.20, random_state=1)


# In[35]:


from imblearn.over_sampling import RandomOverSampler
OS = RandomOverSampler(random_state=15)

X_train_ox, y_train_ox = OS.fit_sample(X_train_tes, y_train_tes)


# In[36]:


sum(y1==0)


# In[37]:


stdsc = StandardScaler()
X_train_std_ox = stdsc.fit_transform(X_train_ox)
X_test_std_ox = stdsc.transform(X_test_ox)


# In[38]:


ox_out_std = stdsc.transform(X1)

plt.figure(figsize=(15,30))

ax = sns.boxplot(data=ox_out_std, orient="h", palette="Set2")


# In[31]:


pca = PCA()
X_train_pca_ox = pca.fit_transform(X_train_std_ox)
pca.explained_variance_ratio_


# In[32]:


plt.bar(np.arange(22), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(np.arange(22), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()


# In[33]:


pca = PCA(n_components=12)
X_train_pca_ox = pca.fit_transform(X_train_std_ox)
pca.explained_variance_ratio_


# In[72]:


svm = SVC(probability=True, verbose=False)

# selecting features
sbs = SBS(svm, k_features=1)
sbs.fit(X_train_std_ox, y_train_ox)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# In[73]:


sbs.subsets_


# In[74]:


forest = RandomForestClassifier()

# selecting features
sbs = SBS(forest, k_features=1)
sbs.fit(X_train_std_ox, y_train_ox)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.2, 1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# In[75]:


sbs.subsets_


# In[76]:


logistR = LogisticRegression()

# selecting features
sbs = SBS(logistR, k_features=1)
sbs.fit(X_train_std_ox, y_train_ox)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# In[77]:


sbs.subsets_


# In[78]:


knn = KNeighborsClassifier(n_neighbors=15)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std_ox, y_train_ox)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# In[79]:


sbs.subsets_


# In[80]:


NaivB = GaussianNB()

# selecting features
sbs = SBS(NaivB, k_features=1)
sbs.fit(X_train_std_ox, y_train_ox)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.5, 1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()


# In[81]:


tpot = TPOTClassifier(generations=5, population_size=20, cv=10,
                                    random_state=42, verbosity=2)

tpot.fit(X_train_std_ox, y_train_ox)


# In[82]:


mlp = MLPClassifier(hidden_layer_sizes=(60,), max_iter=10, solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1)


# In[83]:


mlp.fit(X_train_std_ox, y_train_ox)
print("Training set score: %f" % mlp.score(X_train_std_ox, y_train_ox))
print("Test set score: %f" % mlp.score(X_test_std_ox, y_test_ox))


# In[84]:


mlp = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=200, alpha=1e-4,
                    activation='logistic', verbose=10, tol=1e-4, random_state=1)
mlp.fit(X_train_std_ox, y_train_ox)
print("Training set score: %f" % mlp.score(X_train_std_ox, y_train_ox))
print("Test set score: %f" % mlp.score(X_test_std_ox, y_test_ox))

prediction = mlp.predict(X_test_std_ox)

print(confusion_matrix(y_test_ox,prediction))


# In[85]:


def create_model():
# create model
    model = Sequential()
    model.add(Dense(30, input_dim=22, activation='relu'))
    model.add(Dense(15, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[86]:


#we can increase the epochs
model = KerasClassifier(build_fn=create_model, epochs=90, batch_size=50, verbose=False)

#try this only at home as it takes a long computing time
model_cross_val = cross_val_score(model, X_train_std_ox, y_train_ox, cv=5).mean()

print('Model cross validation score %.5f'% (model_cross_val))


# In[87]:


model.fit(X_train_std_ox, y_train_ox)
y_train_pred = model.predict(X_train_std_ox)
y_test_pred = model.predict(X_test_std_ox)

model_train = accuracy_score(y_train_ox, y_train_pred) 
model_test = accuracy_score(y_test_ox, y_test_pred) 
print('Model train accuracies %.5f'% (model_train))
print('Model test accuracies %.5f'% (model_test))


# In[88]:


pipe1 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=7)], ['SVC', SVC(probability=True, verbose=False)]])
pipe2 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=10)], ['RF', RandomForestClassifier()]])
pipe3 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=6)], ['LoR', LogisticRegression()]])
pipe4 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=5)], ['KNN', KNeighborsClassifier()]])               
pipe5 = Pipeline([['sc', StandardScaler()], ['PCA', PCA(n_components=5)], ['GB', GaussianNB()]])    
pipe6 = Pipeline([['sc', StandardScaler()], ['MLP', MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000, alpha=1e-4,
                    activation='logistic', verbose=10, tol=1e-4, random_state=1)]])

estimators = []
estimators.append(('SVC', pipe1))
estimators.append(('RF', pipe2))
estimators.append(('LoR', pipe3))
estimators.append(('KNN', pipe4))
estimators.append(('GB', pipe5))
estimators.append(('MLP', pipe6))
clf_labels = ['Support Vector Machine', 'Random Forest', 'Logistic Regression', 'KNN','Gaussian NaiveBayes', 'MLP']


# In[89]:


# Hard Voting
ensemble = VotingClassifier(estimators, voting='hard')
all_clf = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, ensemble]
results = model_selection.cross_val_score(ensemble, X=X_train_ox, y=y_train_ox, cv=10)
results


# In[90]:


for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train_ox, y=y_train_ox, cv=10, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))


# In[91]:


# Soft Voting

ensemble = VotingClassifier(estimators, voting='soft')
all_clf = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, ensemble]
results_soft = model_selection.cross_val_score(ensemble, X=X_train_ox, y=y_train_ox, cv=10)
results_soft


# In[92]:


for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train_ox, y=y_train_ox, cv=10, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f) [%s]"  % (scores.mean(), scores.std(), label))


# In[34]:


#dummy
pipe_dc = Pipeline([('pca', PCA(n_components=12)),
                     ('clf', DummyClassifier())])

#Fit Pipeline to training Data
pipe_dc.fit(X_train_std_ox, y_train_ox)

num_folds = 10

scores = cross_val_score(estimator=pipe_dc, X=X_train_std_ox, y=y_train_ox, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))


# In[36]:


#confusiong matrix NB

pipe_dc.fit(X_train_std_ox, y_train_ox)
y_pred_dc = pipe_dc.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_dc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_dc))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_dc))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_dc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_dc))


print(classification_report(y_test_ox, y_pred_dc))


# In[39]:


#Make NB Pipeline
pipe_nb = Pipeline([('pca', PCA(n_components=8)),
                     ('clf', GaussianNB())])

#Fit Pipeline to training Data
pipe_nb.fit(X_train_std_ox, y_train_ox)

num_folds = 10

scores = cross_val_score(estimator=pipe_nb, X=X_train_std_ox, y=y_train_ox, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))


# In[40]:


#confusiong matrix NB

pipe_nb.fit(X_train_std_ox, y_train_ox)
y_pred_nb = pipe_nb.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_nb)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_nb))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_nb))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_nb))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_nb))


print(classification_report(y_test_ox, y_pred_nb))


# In[44]:


#Make Support Vector Classifier Pipeline
pipe_svc = Pipeline([('pca', PCA(n_components=8)),
                     ('clf', SVC(probability=True, verbose=False))])

#Fit Pipeline to training Data
pipe_svc.fit(X_train_std_ox, y_train_ox)

num_folds = 10

scores = cross_val_score(estimator=pipe_svc, X=X_train_std_ox, y=y_train_ox, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},
              {'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]

gs_svc = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_svc = gs_svc.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_svc.best_score_)
print('--> Best Parameters: \n',gs_svc.best_params_)


# In[45]:


#FINALIZE MODEL
#Use best parameters
clf_svc = gs_svc.best_estimator_

#Get Final Scores
clf_svc.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_svc,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_svc.score(X_test_std_ox,y_test_ox))


# In[46]:


#confusiong matrix SVM

clf_svc.fit(X_train_std_ox, y_train_ox)
y_pred_svc = clf_svc.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_svc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_svc))

print(classification_report(y_test_ox, y_pred_svc))


# In[47]:


#Make Random Forest Pipeline

pipe_rf = Pipeline([('pca', PCA(n_components=8)),
                    ('clf', RandomForestClassifier())])

#Fit Pipeline to training Data
pipe_rf.fit(X_train_std_ox, y_train_ox)

num_folds = 10

scores = cross_val_score(estimator=pipe_rf, X=X_train_std_ox, y=y_train_ox, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
params = {'clf__criterion':['gini','entropy'],
          'clf__n_estimators':[10,15,20,25,30],
          'clf__min_samples_leaf':[1,2,3],
          'clf__min_samples_split':[3,4,5,6,7], 
          'clf__random_state':[1]}

gs_rf = GridSearchCV(estimator=pipe_rf,
                  param_grid=params,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=-1)

gs_rf = gs_rf.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_rf.best_score_)
print('--> Best Parameters: \n',gs_rf.best_params_)


# In[48]:


#FINALIZE MODEL
#Use best parameters
clf_rf = gs_rf.best_estimator_

#Get Final Scores
clf_rf.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_rf,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_rf.score(X_test_std_ox,y_test_ox))


# In[49]:


#confusiong matrix RF

clf_rf.fit(X_train_std_ox, y_train_ox)
y_pred_rf = clf_rf.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_rf)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()

print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_rf))


print(classification_report(y_test_ox, y_pred_rf))


# In[50]:


#Make Logistic Regression Classifier Pipeline
pipe_lr = Pipeline([('pca', PCA(n_components=8)),
                     ('clf', LogisticRegression())])

#Fit Pipeline to training Data
pipe_lr.fit(X_train_std_ox, y_train_ox)

num_folds = 10

scores = cross_val_score(estimator=pipe_lr, X=X_train_std_ox, y=y_train_ox, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = {'clf__C': param_range,'clf__penalty': ['l1', 'l2']}

gs_lr = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_lr = gs_lr.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_lr.best_score_)
print('--> Best Parameters: \n',gs_lr.best_params_)


# In[51]:


#FINALIZE MODEL
#Use best parameters
clf_lr = gs_lr.best_estimator_

#Get Final Scores
clf_lr.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_lr,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_lr.score(X_test_std_ox,y_test_ox))


# In[52]:


#confusiong matrix LogR

clf_lr.fit(X_train_std_ox, y_train_ox)
y_pred_lr = clf_lr.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_lr)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_lr))

print(classification_report(y_test_ox, y_pred_lr))


# In[53]:


#Make KNN Classifier Pipeline
pipe_knn = Pipeline([('pca', PCA(n_components=8)),
                     ('clf', KNeighborsClassifier())])

#Fit Pipeline to training Data
pipe_knn.fit(X_train_std_ox, y_train_ox)

num_folds = 10

scores = cross_val_score(estimator=pipe_knn, X=X_train_std_ox, y=y_train_ox, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

param_grid = {'clf__n_neighbors': param_range,'clf__weights': ['uniform', 'distance']}

gs_knn = GridSearchCV(estimator=pipe_knn,
                  param_grid=param_grid,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_knn = gs_knn.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_knn.best_score_)
print('--> Best Parameters: \n',gs_knn.best_params_)


# In[54]:


#FINALIZE MODEL
#Use best parameters
clf_knn = gs_knn.best_estimator_

#Get Final Scores
clf_knn.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_knn,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_knn.score(X_test_std_ox,y_test_ox))


# In[55]:


#confusiong matrix KNN

clf_knn.fit(X_train_std_ox, y_train_ox)
y_pred_knn = clf_knn.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_knn)
print(confmat)



fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_knn))


print(classification_report(y_test_ox, y_pred_knn))


# In[56]:


#Make MLP Classifier Pipeline
pipe_mlp = Pipeline([('clf', MLPClassifier(max_iter=2000))])

#Fit Pipeline to training Data
pipe_mlp.fit(X_train_std_ox, y_train_ox)

num_folds = 10

scores = cross_val_score(estimator=pipe_mlp, X=X_train_std_ox, y=y_train_ox, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters

param_grid = {'clf__solver': ['lbfgs','sgd','adam'],'clf__hidden_layer_sizes': [(40,40),(50,50),(60,60),(70,70)],
              'clf__alpha': [0.001,0.0001],'clf__activation': ['logistic','relu']}

gs_mlp = GridSearchCV(estimator=pipe_mlp,
                  param_grid=param_grid,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_mlp = gs_mlp.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_mlp.best_score_)
print('--> Best Parameters: \n',gs_mlp.best_params_)


# In[57]:


#FINALIZE MODEL
#Use best parameters
clf_mlp = gs_mlp.best_estimator_

#Get Final Scores
clf_mlp.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_mlp,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_mlp.score(X_test_std_ox,y_test_ox))


# In[58]:


#confusiong matrix MLP

clf_mlp.fit(X_train_std_ox, y_train_ox)
y_pred_mlp = clf_mlp.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_mlp)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_mlp))


print(classification_report(y_test_ox, y_pred_mlp))


# In[59]:


tpot = TPOTClassifier(generations=5, population_size=20, cv=10, scoring='accuracy',
                                    random_state=42, verbosity=2)


# In[60]:


#confusiong matrix tpot

tpot.fit(X_train_std_ox, y_train_ox)
y_pred_tpot = tpot.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_tpot)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_tpot))


print(classification_report(y_test_ox, y_pred_tpot))


# In[61]:


def create_model(activation='relu',neurons=10):
# create model
    model = Sequential()
    model.add(Dense(30, input_dim=22, activation='relu'))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[62]:


#Make Keras Classifier Pipeline
pipe_kc = Pipeline([('clf', KerasClassifier(build_fn=create_model, verbose=False))])

#Fit Pipeline to training Data
pipe_kc.fit(X_train_std_ox, y_train_ox)

num_folds = 5

scores = cross_val_score(estimator=pipe_kc, X=X_train_std_ox, y=y_train_ox, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters

param_grid = {'clf__neurons': [8,10,15],'clf__activation': ['sigmoid','relu','tanh'],
              'clf__epochs': [100],'clf__batch_size': [30,50]}

gs_kc = GridSearchCV(estimator=pipe_kc,
                  param_grid=param_grid,
                  scoring='accuracy', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_kc = gs_kc.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_kc.best_score_)
print('--> Best Parameters: \n',gs_kc.best_params_)


# In[63]:


#FINALIZE MODEL
#Use best parameters
clf_kc = gs_kc.best_estimator_


# In[64]:


#confusiong matrix KC

clf_kc.fit(X_train_std_ox, y_train_ox)
y_pred_kc = clf_kc.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_kc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_kc))


print(classification_report(y_test_ox, y_pred_kc))


# In[65]:


#ROC
fig = plt.figure(figsize=(8, 6))
all_tpr = []


probas = clf_svc.predict_proba(X_test_std_ox)
fpr, tpr, thresholds = roc_curve(y_true=y_test_ox, y_score=probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
current_auc = str('%.2f' %roc_auc)

probas_rf = clf_rf.predict_proba(X_test_std_ox)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_true=y_test_ox, y_score=probas_rf[:, 1], pos_label=1)
roc_auc = auc(fpr_rf, tpr_rf)
current_auc = str('%.2f' %roc_auc)

probas_lr = clf_lr.predict_proba(X_test_std_ox)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_true=y_test_ox, y_score=probas_lr[:, 1], pos_label=1)
roc_auc = auc(fpr_lr, tpr_lr)
current_auc = str('%.2f' %roc_auc)

probas_knn = clf_knn.predict_proba(X_test_std_ox)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_true=y_test_ox, y_score=probas_knn[:, 1], pos_label=1)
roc_auc = auc(fpr_knn, tpr_knn)
current_auc = str('%.2f' %roc_auc)

probas_mlp = clf_mlp.predict_proba(X_test_std_ox)
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_true=y_test_ox, y_score=probas_mlp[:, 1], pos_label=1)
roc_auc = auc(fpr_mlp, tpr_mlp)
current_auc = str('%.2f' %roc_auc)

probas_tpot = tpot.predict_proba(X_test_std_ox)
fpr_tpot, tpr_tpot, thresholds_tpot = roc_curve(y_true=y_test_ox, y_score=probas_tpot[:, 1], pos_label=1)
roc_auc = auc(fpr_tpot, tpr_tpot)
current_auc = str('%.2f' %roc_auc)

probas_nb = pipe_nb.predict_proba(X_test_std_ox)
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_true=y_test_ox, y_score=probas_nb[:, 1], pos_label=1)
roc_auc = auc(fpr_nb, tpr_nb)
current_auc = str('%.2f' %roc_auc)

probas_kc = clf_kc.predict_proba(X_test_std_ox)
fpr_kc, tpr_kc, thresholds_kc = roc_curve(y_true=y_test_ox, y_score=probas_kc[:, 1], pos_label=1)
roc_auc = auc(fpr_kc, tpr_kc)
current_auc = str('%.2f' %roc_auc)



plt.plot(fpr, 
         tpr, 
         lw=1,
         label='SVM')

plt.plot(fpr_rf, 
         tpr_rf, 
         lw=1,
         label='Random Forest')

plt.plot(fpr_lr, 
         tpr_lr, 
         lw=1,
         label='Logistic Regression')

plt.plot(fpr_knn, 
         tpr_knn, 
         lw=1,
         label='KNN')

plt.plot(fpr_mlp, 
         tpr_mlp, 
         lw=1,
         label='MLP')

plt.plot(fpr_tpot, 
         tpr_tpot, 
         lw=1,
         label='TPOT')

plt.plot(fpr_kc, 
         tpr_kc, 
         lw=1,
         label='KC')

plt.plot(fpr_nb, 
         tpr_nb, 
         lw=1,
         label='NB')

plt.plot([0, 1], 
         [0, 1], 
         linestyle='--', 
         color=(0.6, 0.6, 0.6), 
         label='random guessing')

plt.plot([0, 0, 1], 
         [0, 1, 1], 
         lw=2, 
         linestyle=':', 
         color='black', 
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[66]:


#SVM Tune Hyperparameters
num_folds = 10

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},
              {'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]

gs_svc = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='f1', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_svc = gs_svc.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_svc.best_score_)
print('--> Best Parameters: \n',gs_svc.best_params_)


# In[67]:


#FINALIZE MODEL
#Use best parameters
clf_svc = gs_svc.best_estimator_

#Get Final Scores
clf_svc.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_svc,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         scoring='f1',
                         cv=num_folds,
                         n_jobs=1)
print('CV f1 scores: %s' % scores)
print('--> Final Model Training f1: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final f1 on Test set: %.5f' % clf_svc.score(X_test_std_ox,y_test_ox))


# In[68]:


#confusiong matrix SVM

clf_svc.fit(X_train_std_ox, y_train_ox)
y_pred_svc = clf_svc.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_svc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_svc))

print(classification_report(y_test_ox, y_pred_svc))


# In[69]:


#Tune Hyperparameters  RF
params = {'clf__criterion':['gini','entropy'],
          'clf__n_estimators':[10,15,20,25,30],
          'clf__min_samples_leaf':[1,2,3],
          'clf__min_samples_split':[3,4,5,6,7], 
          'clf__random_state':[1]}

gs_rf = GridSearchCV(estimator=pipe_rf,
                  param_grid=params,
                  scoring='f1', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=-1)

gs_rf = gs_rf.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_rf.best_score_)
print('--> Best Parameters: \n',gs_rf.best_params_)


# In[70]:


#FINALIZE MODEL
#Use best parameters
clf_rf = gs_rf.best_estimator_

#Get Final Scores
clf_rf.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_rf,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         scoring='f1',
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_rf.score(X_test_std_ox,y_test_ox))


# In[71]:


#confusiong matrix RF

clf_rf.fit(X_train_std_ox, y_train_ox)
y_pred_rf = clf_rf.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_rf)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()

print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_rf))


print(classification_report(y_test_ox, y_pred_rf))


# In[72]:



#Tune Hyperparameters  LR
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = {'clf__C': param_range,'clf__penalty': ['l1', 'l2']}

gs_lr = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid,
                  scoring='f1', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_lr = gs_lr.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_lr.best_score_)
print('--> Best Parameters: \n',gs_lr.best_params_)


# In[73]:


#FINALIZE MODEL
#Use best parameters
clf_lr = gs_lr.best_estimator_

#Get Final Scores
clf_lr.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_lr,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         scoring='f1',
                         cv=num_folds,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_lr.score(X_test_std_ox,y_test_ox))


# In[74]:


#confusiong matrix LogR

clf_lr.fit(X_train_std_ox, y_train_ox)
y_pred_lr = clf_lr.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_lr)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_lr))

print(classification_report(y_test_ox, y_pred_lr))


# In[75]:


#Make KNN Classifier Pipeline

#Tune Hyperparameters
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

param_grid = {'clf__n_neighbors': param_range,'clf__weights': ['uniform', 'distance']}

gs_knn = GridSearchCV(estimator=pipe_knn,
                  param_grid=param_grid,
                  scoring='f1', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_knn = gs_knn.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_knn.best_score_)
print('--> Best Parameters: \n',gs_knn.best_params_)


# In[76]:


#FINALIZE MODEL
#Use best parameters
clf_knn = gs_knn.best_estimator_

#Get Final Scores
clf_knn.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_knn,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='f1',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_knn.score(X_test_std_ox,y_test_ox))


# In[77]:


#confusiong matrix KNN

clf_knn.fit(X_train_std_ox, y_train_ox)
y_pred_knn = clf_knn.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_knn)
print(confmat)



fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_knn))


print(classification_report(y_test_ox, y_pred_knn))


# In[78]:



#Tune Hyperparameters

param_grid = {'clf__solver': ['lbfgs','sgd','adam'],'clf__hidden_layer_sizes': [(40,40),(50,50),(60,60),(70,70)],
              'clf__alpha': [0.001,0.0001],'clf__activation': ['logistic','relu']}

gs_mlp = GridSearchCV(estimator=pipe_mlp,
                  param_grid=param_grid,
                  scoring='f1', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_mlp = gs_mlp.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_mlp.best_score_)
print('--> Best Parameters: \n',gs_mlp.best_params_)


# In[79]:


#FINALIZE MODEL
#Use best parameters
clf_mlp = gs_mlp.best_estimator_

#Get Final Scores
clf_mlp.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_mlp,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='f1',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_mlp.score(X_test_std_ox,y_test_ox))


# In[80]:


#confusiong matrix MLP

clf_mlp.fit(X_train_std_ox, y_train_ox)
y_pred_mlp = clf_mlp.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_mlp)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_mlp))


print(classification_report(y_test_ox, y_pred_mlp))


# In[81]:


tpot = TPOTClassifier(generations=5, population_size=20, cv=10, scoring='f1',
                                    random_state=42, verbosity=2)

#tpot.fit(X_train_std_ox, y_train_ox)


# In[82]:


#tpot.score(X_test_std_ox, y_test_ox)


# In[83]:


#tpot.predict(X_test_std_ox)


# In[84]:


#confusiong matrix MLP

tpot.fit(X_train_std_ox, y_train_ox)
y_pred_tpot = tpot.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_tpot)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_tpot))


print(classification_report(y_test_ox, y_pred_tpot))


# In[85]:


def create_model(activation='relu',neurons=10):
# create model
    model = Sequential()
    model.add(Dense(30, input_dim=22, activation='relu'))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    return model


# In[89]:


#Make Keras Classifier Pipeline
pipe_kc = Pipeline([('clf', KerasClassifier(build_fn=create_model, verbose=False))])

#Fit Pipeline to training Data
pipe_kc.fit(X_train_std_ox, y_train_ox)

num_folds = 5

#Tune Hyperparameters

param_grid = {'clf__neurons': [8,10,15],'clf__activation': ['sigmoid','relu','tanh'],
              'clf__epochs': [100],'clf__batch_size': [30,50]}

gs_kc = GridSearchCV(estimator=pipe_kc,
                  param_grid=param_grid,
                  scoring='f1', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_kc = gs_kc.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_kc.best_score_)
print('--> Best Parameters: \n',gs_kc.best_params_)


# In[90]:


#FINALIZE MODEL
#Use best parameters
clf_kc = gs_kc.best_estimator_


# In[91]:


#confusiong matrix KC

clf_kc.fit(X_train_std_ox, y_train_ox)
y_pred_kc = clf_kc.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_kc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_kc))


print(classification_report(y_test_ox, y_pred_kc))


# In[92]:


#ROC
fig = plt.figure(figsize=(8, 6))
all_tpr = []


probas = clf_svc.predict_proba(X_test_std_ox)
fpr, tpr, thresholds = roc_curve(y_true=y_test_ox, y_score=probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
current_auc = str('%.2f' %roc_auc)

probas_rf = clf_rf.predict_proba(X_test_std_ox)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_true=y_test_ox, y_score=probas_rf[:, 1], pos_label=1)
roc_auc = auc(fpr_rf, tpr_rf)
current_auc = str('%.2f' %roc_auc)

probas_lr = clf_lr.predict_proba(X_test_std_ox)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_true=y_test_ox, y_score=probas_lr[:, 1], pos_label=1)
roc_auc = auc(fpr_lr, tpr_lr)
current_auc = str('%.2f' %roc_auc)

probas_knn = clf_knn.predict_proba(X_test_std_ox)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_true=y_test_ox, y_score=probas_knn[:, 1], pos_label=1)
roc_auc = auc(fpr_knn, tpr_knn)
current_auc = str('%.2f' %roc_auc)

probas_mlp = clf_mlp.predict_proba(X_test_std_ox)
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_true=y_test_ox, y_score=probas_mlp[:, 1], pos_label=1)
roc_auc = auc(fpr_mlp, tpr_mlp)
current_auc = str('%.2f' %roc_auc)

probas_tpot = tpot.predict_proba(X_test_std_ox)
fpr_tpot, tpr_tpot, thresholds_tpot = roc_curve(y_true=y_test_ox, y_score=probas_tpot[:, 1], pos_label=1)
roc_auc = auc(fpr_tpot, tpr_tpot)
current_auc = str('%.2f' %roc_auc)

probas_nb = pipe_nb.predict_proba(X_test_std_ox)
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_true=y_test_ox, y_score=probas_nb[:, 1], pos_label=1)
roc_auc = auc(fpr_nb, tpr_nb)
current_auc = str('%.2f' %roc_auc)

probas_kc = clf_kc.predict_proba(X_test_std_ox)
fpr_kc, tpr_kc, thresholds_kc = roc_curve(y_true=y_test_ox, y_score=probas_kc[:, 1], pos_label=1)
roc_auc = auc(fpr_kc, tpr_kc)
current_auc = str('%.2f' %roc_auc)

plt.plot(fpr, 
         tpr, 
         lw=1,
         label='SVM')

plt.plot(fpr_rf, 
         tpr_rf, 
         lw=1,
         label='Random Forest')

plt.plot(fpr_lr, 
         tpr_lr, 
         lw=1,
         label='Logistic Regression')

plt.plot(fpr_knn, 
         tpr_knn, 
         lw=1,
         label='KNN')

plt.plot(fpr_mlp, 
         tpr_mlp, 
         lw=1,
         label='MLP')

plt.plot(fpr_tpot, 
         tpr_tpot, 
         lw=1,
         label='TPOT')

plt.plot(fpr_kc, 
         tpr_kc, 
         lw=1,
         label='KC')

plt.plot(fpr_nb, 
         tpr_nb, 
         lw=1,
         label='NB')

plt.plot([0, 1], 
         [0, 1], 
         linestyle='--', 
         color=(0.6, 0.6, 0.6), 
         label='random guessing')

plt.plot([0, 0, 1], 
         [0, 1, 1], 
         lw=2, 
         linestyle=':', 
         color='black', 
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[ ]:


#precision


# In[93]:


#SVM Tune Hyperparameters
num_folds = 10

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},
              {'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]

gs_svc = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='precision', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_svc = gs_svc.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_svc.best_score_)
print('--> Best Parameters: \n',gs_svc.best_params_)


# In[94]:


#FINALIZE MODEL
#Use best parameters
clf_svc = gs_svc.best_estimator_

#Get Final Scores
clf_svc.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_svc,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='precision',
                         n_jobs=1)
print('CV precision scores: %s' % scores)
print('--> Final Model Training precision: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final precision on Test set: %.5f' % clf_svc.score(X_test_std_ox,y_test_ox))


# In[95]:


#confusiong matrix SVM

clf_svc.fit(X_train_std_ox, y_train_ox)
y_pred_svc = clf_svc.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_svc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_svc))

print(classification_report(y_test_ox, y_pred_svc))


# In[96]:


#Tune Hyperparameters RF
params = {'clf__criterion':['gini','entropy'],
          'clf__n_estimators':[10,15,20,25,30],
          'clf__min_samples_leaf':[1,2,3],
          'clf__min_samples_split':[3,4,5,6,7], 
          'clf__random_state':[1]}

gs_rf = GridSearchCV(estimator=pipe_rf,
                  param_grid=params,
                  scoring='precision', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=-1)

gs_rf = gs_rf.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_rf.best_score_)
print('--> Best Parameters: \n',gs_rf.best_params_)


# In[97]:


#FINALIZE MODEL
#Use best parameters
clf_rf = gs_rf.best_estimator_

#Get Final Scores
clf_rf.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_rf,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='precision',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_rf.score(X_test_std_ox,y_test_ox))


# In[98]:


#confusiong matrix RF

clf_rf.fit(X_train_std_ox, y_train_ox)
y_pred_rf = clf_rf.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_rf)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()

print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_rf))


print(classification_report(y_test_ox, y_pred_rf))


# In[99]:



#Tune Hyperparameters LR
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = {'clf__C': param_range,'clf__penalty': ['l1', 'l2']}

gs_lr = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid,
                  scoring='precision', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_lr = gs_lr.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_lr.best_score_)
print('--> Best Parameters: \n',gs_lr.best_params_)


# In[100]:


#FINALIZE MODEL
#Use best parameters
clf_lr = gs_lr.best_estimator_

#Get Final Scores
clf_lr.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_lr,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='precision',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_lr.score(X_test_std_ox,y_test_ox))


# In[101]:


#confusiong matrix LogR

clf_lr.fit(X_train_std_ox, y_train_ox)
y_pred_lr = clf_lr.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_lr)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_lr))

print(classification_report(y_test_ox, y_pred_lr))


# In[102]:


#Make KNN Classifier Pipeline
pipe_knn = Pipeline([('pca', PCA(n_components=20)),
                     ('clf', KNeighborsClassifier())])

#Fit Pipeline to training Data
pipe_knn.fit(X_train_std_ox, y_train_ox)

num_folds = 10

scores = cross_val_score(estimator=pipe_knn, X=X_train_std_ox, y=y_train_ox, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

param_grid = {'clf__n_neighbors': param_range,'clf__weights': ['uniform', 'distance']}

gs_knn = GridSearchCV(estimator=pipe_knn,
                  param_grid=param_grid,
                  scoring='precision', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_knn = gs_knn.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_knn.best_score_)
print('--> Best Parameters: \n',gs_knn.best_params_)


# In[103]:


#FINALIZE MODEL
#Use best parameters
clf_knn = gs_knn.best_estimator_

#Get Final Scores
clf_knn.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_knn,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='precision',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_knn.score(X_test_std_ox,y_test_ox))


# In[104]:


#confusiong matrix KNN

clf_knn.fit(X_train_std_ox, y_train_ox)
y_pred_knn = clf_knn.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_knn)
print(confmat)



fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_knn))


print(classification_report(y_test_ox, y_pred_knn))


# In[105]:



#Tune Hyperparameters

param_grid = {'clf__solver': ['lbfgs','sgd','adam'],'clf__hidden_layer_sizes': [(40,40),(50,50),(60,60),(70,70)],
              'clf__alpha': [0.001,0.0001],'clf__activation': ['logistic','relu']}

gs_mlp = GridSearchCV(estimator=pipe_mlp,
                  param_grid=param_grid,
                  scoring='precision', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_mlp = gs_mlp.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_mlp.best_score_)
print('--> Best Parameters: \n',gs_mlp.best_params_)


# In[106]:


#FINALIZE MODEL
#Use best parameters
clf_mlp = gs_mlp.best_estimator_

#Get Final Scores
clf_mlp.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_mlp,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='precision',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_mlp.score(X_test_std_ox,y_test_ox))


# In[107]:


#confusiong matrix MLP

clf_mlp.fit(X_train_std_ox, y_train_ox)
y_pred_mlp = clf_mlp.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_mlp)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_mlp))


print(classification_report(y_test_ox, y_pred_mlp))


# In[108]:


tpot = TPOTClassifier(generations=5, population_size=20, cv=10, scoring='precision',
                                    random_state=42, verbosity=2)


# In[109]:


#confusiong matrix TPOT

tpot.fit(X_train_std_ox, y_train_ox)
y_pred_tpot = tpot.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_tpot)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_tpot))


print(classification_report(y_test_ox, y_pred_tpot))


# In[110]:


def create_model(activation='relu',neurons=10):
# create model
    model = Sequential()
    model.add(Dense(30, input_dim=22, activation='relu'))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[precision])
    return model


# In[111]:


#Make Keras Classifier Pipeline
pipe_kc = Pipeline([('clf', KerasClassifier(build_fn=create_model, verbose=False))])

#Fit Pipeline to training Data
pipe_kc.fit(X_train_std_ox, y_train_ox)

num_folds = 5

#Tune Hyperparameters

param_grid = {'clf__neurons': [8,10,15],'clf__activation': ['sigmoid','relu','tanh'],
              'clf__epochs': [100],'clf__batch_size': [30,50]}

gs_kc = GridSearchCV(estimator=pipe_kc,
                  param_grid=param_grid,
                  scoring='precision', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_kc = gs_kc.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_kc.best_score_)
print('--> Best Parameters: \n',gs_kc.best_params_)


# In[112]:


#FINALIZE MODEL
#Use best parameters
clf_kc = gs_kc.best_estimator_


# In[113]:


#confusiong matrix KC

clf_kc.fit(X_train_std_ox, y_train_ox)
y_pred_kc = clf_kc.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_kc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_kc))


print(classification_report(y_test_ox, y_pred_kc))


# In[114]:


#ROC
fig = plt.figure(figsize=(8, 6))
all_tpr = []


probas = clf_svc.predict_proba(X_test_std_ox)
fpr, tpr, thresholds = roc_curve(y_true=y_test_ox, y_score=probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
current_auc = str('%.2f' %roc_auc)

probas_rf = clf_rf.predict_proba(X_test_std_ox)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_true=y_test_ox, y_score=probas_rf[:, 1], pos_label=1)
roc_auc = auc(fpr_rf, tpr_rf)
current_auc = str('%.2f' %roc_auc)

probas_lr = clf_lr.predict_proba(X_test_std_ox)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_true=y_test_ox, y_score=probas_lr[:, 1], pos_label=1)
roc_auc = auc(fpr_lr, tpr_lr)
current_auc = str('%.2f' %roc_auc)

probas_knn = clf_knn.predict_proba(X_test_std_ox)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_true=y_test_ox, y_score=probas_knn[:, 1], pos_label=1)
roc_auc = auc(fpr_knn, tpr_knn)
current_auc = str('%.2f' %roc_auc)

probas_mlp = clf_mlp.predict_proba(X_test_std_ox)
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_true=y_test_ox, y_score=probas_mlp[:, 1], pos_label=1)
roc_auc = auc(fpr_mlp, tpr_mlp)
current_auc = str('%.2f' %roc_auc)

probas_tpot = tpot.predict_proba(X_test_std_ox)
fpr_tpot, tpr_tpot, thresholds_tpot = roc_curve(y_true=y_test_ox, y_score=probas_tpot[:, 1], pos_label=1)
roc_auc = auc(fpr_tpot, tpr_tpot)
current_auc = str('%.2f' %roc_auc)

probas_nb = pipe_nb.predict_proba(X_test_std_ox)
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_true=y_test_ox, y_score=probas_nb[:, 1], pos_label=1)
roc_auc = auc(fpr_nb, tpr_nb)
current_auc = str('%.2f' %roc_auc)

probas_kc = clf_kc.predict_proba(X_test_std_ox)
fpr_kc, tpr_kc, thresholds_kc = roc_curve(y_true=y_test_ox, y_score=probas_kc[:, 1], pos_label=1)
roc_auc = auc(fpr_kc, tpr_kc)
current_auc = str('%.2f' %roc_auc)


plt.plot(fpr, 
         tpr, 
         lw=1,
         label='SVM')

plt.plot(fpr_rf, 
         tpr_rf, 
         lw=1,
         label='Random Forest')

plt.plot(fpr_lr, 
         tpr_lr, 
         lw=1,
         label='Logistic Regression')

plt.plot(fpr_knn, 
         tpr_knn, 
         lw=1,
         label='KNN')

plt.plot(fpr_mlp, 
         tpr_mlp, 
         lw=1,
         label='MLP')

plt.plot(fpr_tpot, 
         tpr_tpot, 
         lw=1,
         label='TPOT')

plt.plot(fpr_kc, 
         tpr_kc, 
         lw=1,
         label='KC')

plt.plot(fpr_nb, 
         tpr_nb, 
         lw=1,
         label='NB')

plt.plot([0, 1], 
         [0, 1], 
         linestyle='--', 
         color=(0.6, 0.6, 0.6), 
         label='random guessing')

plt.plot([0, 0, 1], 
         [0, 1, 1], 
         lw=2, 
         linestyle=':', 
         color='black', 
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[ ]:


#recall


# In[115]:


#SVM Tune Hyperparameters
num_folds = 10

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},
              {'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]

gs_svc = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='recall', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_svc = gs_svc.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_svc.best_score_)
print('--> Best Parameters: \n',gs_svc.best_params_)


# In[116]:


#FINALIZE MODEL
#Use best parameters
clf_svc = gs_svc.best_estimator_

#Get Final Scores
clf_svc.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_svc,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='recall',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_svc.score(X_test_std_ox,y_test_ox))


# In[117]:


#confusiong matrix SVM

clf_svc.fit(X_train_std_ox, y_train_ox)
y_pred_svc = clf_svc.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_svc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_svc))

print(classification_report(y_test_ox, y_pred_svc))


# In[118]:


#Tune Hyperparameters
params = {'clf__criterion':['gini','entropy'],
          'clf__n_estimators':[10,15,20,25,30],
          'clf__min_samples_leaf':[1,2,3],
          'clf__min_samples_split':[3,4,5,6,7], 
          'clf__random_state':[1]}

gs_rf = GridSearchCV(estimator=pipe_rf,
                  param_grid=params,
                  scoring='recall', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=-1)

gs_rf = gs_rf.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_rf.best_score_)
print('--> Best Parameters: \n',gs_rf.best_params_)


# In[119]:


#FINALIZE MODEL
#Use best parameters
clf_rf = gs_rf.best_estimator_

#Get Final Scores
clf_rf.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_rf,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='recall',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_rf.score(X_test_std_ox,y_test_ox))


# In[120]:


#confusiong matrix RF

clf_rf.fit(X_train_std_ox, y_train_ox)
y_pred_rf = clf_rf.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_rf)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()

print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_rf))


print(classification_report(y_test_ox, y_pred_rf))


# In[121]:



#Tune Hyperparameters  LR
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = {'clf__C': param_range,'clf__penalty': ['l1', 'l2']}

gs_lr = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid,
                  scoring='recall', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_lr = gs_lr.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_lr.best_score_)
print('--> Best Parameters: \n',gs_lr.best_params_)


# In[122]:


#FINALIZE MODEL
#Use best parameters
clf_lr = gs_lr.best_estimator_

#Get Final Scores
clf_lr.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_lr,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='recall',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_lr.score(X_test_std_ox,y_test_ox))


# In[123]:


#confusiong matrix LogR

clf_lr.fit(X_train_std_ox, y_train_ox)
y_pred_lr = clf_lr.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_lr)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_lr))

print(classification_report(y_test_ox, y_pred_lr))


# In[124]:


#Make KNN Classifier Pipeline
pipe_knn = Pipeline([('pca', PCA(n_components=20)),
                     ('clf', KNeighborsClassifier())])

#Fit Pipeline to training Data
pipe_knn.fit(X_train_std_ox, y_train_ox)

num_folds = 10

scores = cross_val_score(estimator=pipe_knn, X=X_train_std_ox, y=y_train_ox, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

param_grid = {'clf__n_neighbors': param_range,'clf__weights': ['uniform', 'distance']}

gs_knn = GridSearchCV(estimator=pipe_knn,
                  param_grid=param_grid,
                  scoring='recall', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_knn = gs_knn.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_knn.best_score_)
print('--> Best Parameters: \n',gs_knn.best_params_)


# In[125]:


#FINALIZE MODEL
#Use best parameters
clf_knn = gs_knn.best_estimator_

#Get Final Scores
clf_knn.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_knn,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='recall',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_knn.score(X_test_std_ox,y_test_ox))


# In[126]:


#confusiong matrix KNN

clf_knn.fit(X_train_std_ox, y_train_ox)
y_pred_knn = clf_knn.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_knn)
print(confmat)



fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_knn))


print(classification_report(y_test_ox, y_pred_knn))


# In[127]:



#Tune Hyperparameters

param_grid = {'clf__solver': ['lbfgs','sgd','adam'],'clf__hidden_layer_sizes': [(40,40),(50,50),(60,60),(70,70)],
              'clf__alpha': [0.001,0.0001],'clf__activation': ['logistic','relu']}

gs_mlp = GridSearchCV(estimator=pipe_mlp,
                  param_grid=param_grid,
                  scoring='recall', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_mlp = gs_mlp.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_mlp.best_score_)
print('--> Best Parameters: \n',gs_mlp.best_params_)


# In[128]:


#FINALIZE MODEL
#Use best parameters
clf_mlp = gs_mlp.best_estimator_

#Get Final Scores
clf_mlp.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_mlp,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='recall',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_mlp.score(X_test_std_ox,y_test_ox))


# In[129]:


#confusiong matrix MLP

clf_mlp.fit(X_train_std_ox, y_train_ox)
y_pred_mlp = clf_mlp.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_mlp)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_mlp))


print(classification_report(y_test_ox, y_pred_mlp))


# In[130]:


tpot = TPOTClassifier(generations=5, population_size=20, cv=10, scoring='recall',
                                    random_state=42, verbosity=2)


# In[131]:


#confusiong matrix TPOT

tpot.fit(X_train_std_ox, y_train_ox)
y_pred_tpot = tpot.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_tpot)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_tpot))


print(classification_report(y_test_ox, y_pred_tpot))


# In[132]:


def create_model(activation='relu',neurons=10):
# create model
    model = Sequential()
    model.add(Dense(30, input_dim=22, activation='relu'))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[recall])
    return model


# In[133]:


#Make Keras Classifier Pipeline
pipe_kc = Pipeline([('clf', KerasClassifier(build_fn=create_model, verbose=False))])

#Fit Pipeline to training Data
pipe_kc.fit(X_train_std_ox, y_train_ox)

num_folds = 5


#Tune Hyperparameters

param_grid = {'clf__neurons': [8,10,15],'clf__activation': ['sigmoid','relu','tanh'],
              'clf__epochs': [100],'clf__batch_size': [30,50]}

gs_kc = GridSearchCV(estimator=pipe_kc,
                  param_grid=param_grid,
                  scoring='recall', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_kc = gs_kc.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_kc.best_score_)
print('--> Best Parameters: \n',gs_kc.best_params_)


# In[134]:


#FINALIZE MODEL
#Use best parameters
clf_kc = gs_kc.best_estimator_



# In[135]:


#confusiong matrix KC

clf_kc.fit(X_train_std_ox, y_train_ox)
y_pred_kc = clf_kc.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_kc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_kc))


print(classification_report(y_test_ox, y_pred_kc))


# In[136]:


#ROC
fig = plt.figure(figsize=(8, 6))
all_tpr = []


probas = clf_svc.predict_proba(X_test_std_ox)
fpr, tpr, thresholds = roc_curve(y_true=y_test_ox, y_score=probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
current_auc = str('%.2f' %roc_auc)

probas_rf = clf_rf.predict_proba(X_test_std_ox)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_true=y_test_ox, y_score=probas_rf[:, 1], pos_label=1)
roc_auc = auc(fpr_rf, tpr_rf)
current_auc = str('%.2f' %roc_auc)

probas_lr = clf_lr.predict_proba(X_test_std_ox)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_true=y_test_ox, y_score=probas_lr[:, 1], pos_label=1)
roc_auc = auc(fpr_lr, tpr_lr)
current_auc = str('%.2f' %roc_auc)

probas_knn = clf_knn.predict_proba(X_test_std_ox)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_true=y_test_ox, y_score=probas_knn[:, 1], pos_label=1)
roc_auc = auc(fpr_knn, tpr_knn)
current_auc = str('%.2f' %roc_auc)

probas_mlp = clf_mlp.predict_proba(X_test_std_ox)
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_true=y_test_ox, y_score=probas_mlp[:, 1], pos_label=1)
roc_auc = auc(fpr_mlp, tpr_mlp)
current_auc = str('%.2f' %roc_auc)

probas_tpot = tpot.predict_proba(X_test_std_ox)
fpr_tpot, tpr_tpot, thresholds_tpot = roc_curve(y_true=y_test_ox, y_score=probas_tpot[:, 1], pos_label=1)
roc_auc = auc(fpr_tpot, tpr_tpot)
current_auc = str('%.2f' %roc_auc)

probas_nb = pipe_nb.predict_proba(X_test_std_ox)
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_true=y_test_ox, y_score=probas_nb[:, 1], pos_label=1)
roc_auc = auc(fpr_nb, tpr_nb)
current_auc = str('%.2f' %roc_auc)

probas_kc = clf_kc.predict_proba(X_test_std_ox)
fpr_kc, tpr_kc, thresholds_kc = roc_curve(y_true=y_test_ox, y_score=probas_kc[:, 1], pos_label=1)
roc_auc = auc(fpr_kc, tpr_kc)
current_auc = str('%.2f' %roc_auc)


plt.plot(fpr, 
         tpr, 
         lw=1,
         label='SVM')

plt.plot(fpr_rf, 
         tpr_rf, 
         lw=1,
         label='Random Forest')

plt.plot(fpr_lr, 
         tpr_lr, 
         lw=1,
         label='Logistic Regression')

plt.plot(fpr_knn, 
         tpr_knn, 
         lw=1,
         label='KNN')

plt.plot(fpr_mlp, 
         tpr_mlp, 
         lw=1,
         label='MLP')

plt.plot(fpr_tpot, 
         tpr_tpot, 
         lw=1,
         label='TPOT')

plt.plot(fpr_kc, 
         tpr_kc, 
         lw=1,
         label='KC')

plt.plot(fpr_nb, 
         tpr_nb, 
         lw=1,
         label='NB')

plt.plot([0, 1], 
         [0, 1], 
         linestyle='--', 
         color=(0.6, 0.6, 0.6), 
         label='random guessing')

plt.plot([0, 0, 1], 
         [0, 1, 1], 
         lw=2, 
         linestyle=':', 
         color='black', 
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[ ]:


#AUC


# In[137]:


#SVM Tune Hyperparameters
num_folds = 10

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},
              {'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]

gs_svc = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='roc_auc', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_svc = gs_svc.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_svc.best_score_)
print('--> Best Parameters: \n',gs_svc.best_params_)


# In[138]:


#FINALIZE MODEL
#Use best parameters
clf_svc = gs_svc.best_estimator_

#Get Final Scores
clf_svc.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_svc,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='roc_auc',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_svc.score(X_test_std_ox,y_test_ox))


# In[139]:


#confusiong matrix SVM

clf_svc.fit(X_train_std_ox, y_train_ox)
y_pred_svc = clf_svc.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_svc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_svc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_svc))

print(classification_report(y_test_ox, y_pred_svc))


# In[140]:


#Tune Hyperparameters
params = {'clf__criterion':['gini','entropy'],
          'clf__n_estimators':[10,15,20,25,30],
          'clf__min_samples_leaf':[1,2,3],
          'clf__min_samples_split':[3,4,5,6,7], 
          'clf__random_state':[1]}

gs_rf = GridSearchCV(estimator=pipe_rf,
                  param_grid=params,
                  scoring='roc_auc', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=-1)

gs_rf = gs_rf.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_rf.best_score_)
print('--> Best Parameters: \n',gs_rf.best_params_)


# In[141]:


#FINALIZE MODEL
#Use best parameters
clf_rf = gs_rf.best_estimator_

#Get Final Scores
clf_rf.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_rf,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='roc_auc',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_rf.score(X_test_std_ox,y_test_ox))


# In[142]:


#confusiong matrix RF

clf_rf.fit(X_train_std_ox, y_train_ox)
y_pred_rf = clf_rf.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_rf)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()

print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_rf))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_rf))


print(classification_report(y_test_ox, y_pred_rf))


# In[143]:



#Tune Hyperparameters
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = {'clf__C': param_range,'clf__penalty': ['l1', 'l2']}

gs_lr = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid,
                  scoring='roc_auc', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_lr = gs_lr.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_lr.best_score_)
print('--> Best Parameters: \n',gs_lr.best_params_)


# In[144]:


#FINALIZE MODEL
#Use best parameters
clf_lr = gs_lr.best_estimator_

#Get Final Scores
clf_lr.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_lr,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='roc_auc',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_lr.score(X_test_std_ox,y_test_ox))


# In[145]:


#confusiong matrix LogR

clf_lr.fit(X_train_std_ox, y_train_ox)
y_pred_lr = clf_lr.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_lr)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_lr))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_lr))

print(classification_report(y_test_ox, y_pred_lr))


# In[146]:


#Make KNN Classifier Pipeline
pipe_knn = Pipeline([('pca', PCA(n_components=20)),
                     ('clf', KNeighborsClassifier())])

#Fit Pipeline to training Data
pipe_knn.fit(X_train_std_ox, y_train_ox)

num_folds = 10

scores = cross_val_score(estimator=pipe_knn, X=X_train_std_ox, y=y_train_ox, cv=num_folds, n_jobs=1, verbose=0)
print('CV accuracy scores: %s' % scores)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

param_grid = {'clf__n_neighbors': param_range,'clf__weights': ['uniform', 'distance']}

gs_knn = GridSearchCV(estimator=pipe_knn,
                  param_grid=param_grid,
                  scoring='roc_auc', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)
gs_knn = gs_knn.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_knn.best_score_)
print('--> Best Parameters: \n',gs_knn.best_params_)


# In[147]:


#FINALIZE MODEL
#Use best parameters
clf_knn = gs_knn.best_estimator_

#Get Final Scores
clf_knn.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_knn,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='roc_auc',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_knn.score(X_test_std_ox,y_test_ox))


# In[148]:


#confusiong matrix KNN

clf_knn.fit(X_train_std_ox, y_train_ox)
y_pred_knn = clf_knn.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_knn)
print(confmat)



fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_knn))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_knn))


print(classification_report(y_test_ox, y_pred_knn))


# In[149]:



#Tune Hyperparameters

param_grid = {'clf__solver': ['lbfgs','sgd','adam'],'clf__hidden_layer_sizes': [(40,40),(50,50),(60,60),(70,70)],
              'clf__alpha': [0.001,0.0001],'clf__activation': ['logistic','relu']}

gs_mlp = GridSearchCV(estimator=pipe_mlp,
                  param_grid=param_grid,
                  scoring='roc_auc', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_mlp = gs_mlp.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_mlp.best_score_)
print('--> Best Parameters: \n',gs_mlp.best_params_)


# In[150]:


#FINALIZE MODEL
#Use best parameters
clf_mlp = gs_mlp.best_estimator_

#Get Final Scores
clf_mlp.fit(X_train_std_ox, y_train_ox)
scores = cross_val_score(estimator=clf_mlp,
                         X=X_train_std_ox,
                         y=y_train_ox,
                         cv=num_folds,
                         scoring='roc_auc',
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('--> Final Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_mlp.score(X_test_std_ox,y_test_ox))


# In[151]:


#confusiong matrix MLP

clf_mlp.fit(X_train_std_ox, y_train_ox)
y_pred_mlp = clf_mlp.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_mlp)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_mlp))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_mlp))


print(classification_report(y_test_ox, y_pred_mlp))


# In[152]:


tpot = TPOTClassifier(generations=5, population_size=20, cv=10, scoring='roc_auc',
                                    random_state=42, verbosity=2)


# In[153]:


#confusiong matrix MLP

tpot.fit(X_train_std_ox, y_train_ox)
y_pred_tpot = tpot.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_tpot)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_tpot))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_tpot))


print(classification_report(y_test_ox, y_pred_tpot))


# In[154]:


def create_model(activation='relu',neurons=10):
# create model
    model = Sequential()
    model.add(Dense(30, input_dim=22, activation='relu'))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_ro])
    return model


# In[161]:


#Make Keras Classifier Pipeline
pipe_kc = Pipeline([('clf', KerasClassifier(build_fn=create_model, verbose=False))])

#Fit Pipeline to training Data
pipe_kc.fit(X_train_std_ox, y_train_ox)

num_folds = 5

#Tune Hyperparameters

param_grid = {'clf__neurons': [8,10,15],'clf__activation': ['sigmoid','relu','tanh'],
              'clf__epochs': [100],'clf__batch_size': [30,50]}

gs_kc = GridSearchCV(estimator=pipe_kc,
                  param_grid=param_grid,
                  scoring='roc_auc', #roc_auc, f1, etc which can be founded in the document.
                  cv=num_folds,
                  n_jobs=1)

gs_kc = gs_kc.fit(X_train_std_ox, y_train_ox)
print('--> Tuned Parameters Best Score: ',gs_kc.best_score_)
print('--> Best Parameters: \n',gs_kc.best_params_)


# In[162]:


#FINALIZE MODEL
#Use best parameters
clf_kc = gs_kc.best_estimator_



# In[163]:


#confusiong matrix KC

clf_kc.fit(X_train_std_ox, y_train_ox)
y_pred_kc = clf_kc.predict(X_test_std_ox)
confmat = confusion_matrix(y_true=y_test_ox, y_pred=y_pred_kc)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
plt.show()


print('Precision: %.3f' % precision_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('Recall: %.3f' % recall_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('F1: %.3f' % f1_score(y_true=y_test_ox, y_pred=y_pred_kc))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test_ox, y_pred=y_pred_kc))


print(classification_report(y_test_ox, y_pred_kc))


# In[164]:


#ROC
fig = plt.figure(figsize=(8, 6))
all_tpr = []


probas = clf_svc.predict_proba(X_test_std_ox)
fpr, tpr, thresholds = roc_curve(y_true=y_test_ox, y_score=probas[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
current_auc = str('%.2f' %roc_auc)

probas_rf = clf_rf.predict_proba(X_test_std_ox)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_true=y_test_ox, y_score=probas_rf[:, 1], pos_label=1)
roc_auc = auc(fpr_rf, tpr_rf)
current_auc = str('%.2f' %roc_auc)

probas_lr = clf_lr.predict_proba(X_test_std_ox)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_true=y_test_ox, y_score=probas_lr[:, 1], pos_label=1)
roc_auc = auc(fpr_lr, tpr_lr)
current_auc = str('%.2f' %roc_auc)

probas_knn = clf_knn.predict_proba(X_test_std_ox)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_true=y_test_ox, y_score=probas_knn[:, 1], pos_label=1)
roc_auc = auc(fpr_knn, tpr_knn)
current_auc = str('%.2f' %roc_auc)

probas_mlp = clf_mlp.predict_proba(X_test_std_ox)
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_true=y_test_ox, y_score=probas_mlp[:, 1], pos_label=1)
roc_auc = auc(fpr_mlp, tpr_mlp)
current_auc = str('%.2f' %roc_auc)

probas_tpot = tpot.predict_proba(X_test_std_ox)
fpr_tpot, tpr_tpot, thresholds_tpot = roc_curve(y_true=y_test_ox, y_score=probas_tpot[:, 1], pos_label=1)
roc_auc = auc(fpr_tpot, tpr_tpot)
current_auc = str('%.2f' %roc_auc)

probas_nb = pipe_nb.predict_proba(X_test_std_ox)
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_true=y_test_ox, y_score=probas_nb[:, 1], pos_label=1)
roc_auc = auc(fpr_nb, tpr_nb)
current_auc = str('%.2f' %roc_auc)

probas_kc = clf_kc.predict_proba(X_test_std_ox)
fpr_kc, tpr_kc, thresholds_kc = roc_curve(y_true=y_test_ox, y_score=probas_kc[:, 1], pos_label=1)
roc_auc = auc(fpr_kc, tpr_kc)
current_auc = str('%.2f' %roc_auc)


plt.plot(fpr, 
         tpr, 
         lw=1,
         label='SVM')

plt.plot(fpr_rf, 
         tpr_rf, 
         lw=1,
         label='Random Forest')

plt.plot(fpr_lr, 
         tpr_lr, 
         lw=1,
         label='Logistic Regression')

plt.plot(fpr_knn, 
         tpr_knn, 
         lw=1,
         label='KNN')

plt.plot(fpr_mlp, 
         tpr_mlp, 
         lw=1,
         label='MLP')

plt.plot(fpr_tpot, 
         tpr_tpot, 
         lw=1,
         label='TPOT')

plt.plot(fpr_kc, 
         tpr_kc, 
         lw=1,
         label='KC')

plt.plot(fpr_nb, 
         tpr_nb, 
         lw=1,
         label='NB')

plt.plot([0, 1], 
         [0, 1], 
         linestyle='--', 
         color=(0.6, 0.6, 0.6), 
         label='random guessing')

plt.plot([0, 0, 1], 
         [0, 1, 1], 
         lw=2, 
         linestyle=':', 
         color='black', 
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

