# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"heading_collapsed": true, "cell_type": "markdown"}
# # imports

# + {"hidden": true}
# %pylab
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, r2_score

# + {"hidden": true}
# %ls

# + {"hidden": true}
wine_data = pd.read_csv('winequality-red.csv')
wine_data.info()

# + {"heading_collapsed": true, "cell_type": "markdown"}
# # basic data inspection

# + {"hidden": true}
wine_data.describe()

# + {"hidden": true}
wine_data.head()

# + {"hidden": true, "cell_type": "markdown"}
# chance level ~42%, highly unbalanced data!

# + {"hidden": true}
figure()
sns.barplot(x=wine_data['quality'], y=wine_data['quality'],  estimator=lambda i: len(i) / float(len(wine_data['quality'])) * 100)
ylabel("Percent")
show()

# + {"hidden": true}
sns.pairplot(wine_data)

# + {"hidden": true}
f, ax = subplots(figsize=(15, 12))
corr = wine_data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True,
            square=True, ax=ax)

# + {"hidden": true, "cell_type": "markdown"}
# look who's significantly *positively* correlated with quality!

# + {"heading_collapsed": true, "cell_type": "markdown"}
# # generate training/test data

# + {"hidden": true}
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# + {"hidden": true}
wine_data.columns

# + {"hidden": true}
X, y = wine_data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']], wine_data['quality']
X.shape, y.shape

# + {"hidden": true}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify = y)

# + {"heading_collapsed": true, "cell_type": "markdown"}
# # linear regression

# + {"hidden": true}
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
predicted = lin_reg.predict(X_test)
lin_reg.score(X_test, y_test) #returns coefficient of determination

# + {"hidden": true}
f, ax = subplots(1, 2, figsize=(11, 6))
ax = subplot(121)
scatter(y_test, predicted)
plot(range(3,9), range(3,9))
ax.set_title('test data')
ax.set_xlabel('true')
ax.set_ylabel('predicted')
ax = subplot(122)
ax.scatter(y_train, lin_reg.predict(X_train), c = 'red')
plot(range(3,9), range(3,9), c = 'r')
ax.set_title('training data')
ax.set_xlabel('true')
ax.set_ylabel('predicted')
tight_layout()
show()

# + {"hidden": true, "cell_type": "markdown"}
# **regression to the mean**; most predictions are around 5.5

# + {"hidden": true}
print('accuracy: ',accuracy_score(np.round(lin_reg.predict(X_train)), y_train))
print('precision, recall, f1: ',precision_recall_fscore_support(np.round(lin_reg.predict(X_train)), y_train))

# + {"heading_collapsed": true, "cell_type": "markdown"}
# # SVM

# + {"hidden": true}
from sklearn import svm

# + {"hidden": true}
clf = svm.SVC(C = 2, gamma='scale', kernel = 'poly', degree = 3)
clf.fit(X_train, y_train)

# + {"hidden": true}
predicted_test = clf.predict(X_test)
predicted_train = clf.predict(X_train)

# + {"hidden": true}
print("accuracy on test data: " ,clf.score(X_test, y_test))
print('R^2 on test data: ', r2_score(y_test, predicted_test))

# + {"hidden": true, "cell_type": "markdown"}
# different kernels and values of C did not improve the prediction

# + {"heading_collapsed": true, "cell_type": "markdown"}
# # LDA/QDA

# + {"hidden": true}
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

# + {"hidden": true}
clf = LinearDiscriminantAnalysis() #QuadraticDiscriminantAnalysis()
clf.fit(X_train, y_train)

# + {"hidden": true}
predicted_test = clf.predict(X_test)
predicted_train = clf.predict(X_train)

# + {"hidden": true}
print("accuracy on test data: " ,clf.score(X_test, y_test))
print('R^2 on test data: ', r2_score(y_test, predicted_test))
# -

# # decision trees

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ## simple tree

# + {"hidden": true}
from sklearn import tree
import graphviz
from sklearn import tree

# + {"hidden": true}
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# + {"hidden": true}
predicted_test = clf.predict(X_test)
predicted_train = clf.predict(X_train)

# + {"hidden": true}
print("accuracy on test data: " ,clf.score(X_test, y_test))
print('R^2 on test data: ', r2_score(y_test, predicted_test))

# + {"hidden": true, "cell_type": "markdown"}
# **feature importances**

# + {"hidden": true}
feature_importance = pd.Series(clf.feature_importances_, wine_data.columns[:-1])
feature_importance.sort_values(ascending=False)

# + {"hidden": true}
f, ax = subplots(1, 2, figsize=(11, 6))
ax = subplot(121)
scatter(y_test, predicted)
plot(range(3,9), range(3,9))
ax.set_title('test data')
ax = subplot(122)
ax.scatter(y_train, predicted_train, c = 'red')
plot(range(3,9), range(3,9), c = 'r')
ax.set_title('training data')
tight_layout()
show()

# + {"hidden": true, "cell_type": "markdown"}
# that's what we call overfitting

# + {"hidden": true, "cell_type": "markdown"}
# ### plot the tree

# + {"hidden": true}
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("red-wine-tree") 

# + {"hidden": true}
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names= ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol'],  
                     class_names= None,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("red-wine-tree") 
graph 

# + {"hidden": true, "cell_type": "markdown"}
# ### changing depth

# + {"hidden": true}
score_test, score_train = [], []
for d in range(1,22):
    clf = tree.DecisionTreeClassifier(max_depth=d)
    clf = clf.fit(X_train, y_train)
    score_test.append(clf.score(X_test, y_test))
    score_train.append(clf.score(X_train, y_train))
f, ax = subplots(1, 1, figsize=(11, 6))
ax = subplot(111)
plot(range(1,22), score_test, label='test')
plot(range(1,22), score_train, label='training')
xlabel('max depth')
ylabel('accuracy')
legend()
show()
#     print('max depth: ', d)
#     print("accuracy on test data: " ,clf.score(X_test, y_test))
#     print('$R^2$ on test data: ', r2_score(y_test, predicted_test))
# -

# ## random forest

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=50, max_depth=4)
clf = clf.fit(X_train, y_train)

feature_importance = pd.Series(clf.feature_importances_, wine_data.columns[:-1])
feature_importance.sort_values(ascending=False)

predicted_test = clf.predict(X_test)
predicted_train = clf.predict(X_train)
print("accuracy on test data: " ,clf.score(X_test, y_test))
print(r'R^2 on test data: ', r2_score(y_test, predicted_test))

max_score = 0
f, ax = subplots(1, 1, figsize=(11, 6))
ax = subplot(111)
for estim in range(10,500,50):
    score_test, score_train = [], []
    for d in range(1,22):
        clf = RandomForestClassifier(n_estimators=estim, max_depth=d, )
        clf = clf.fit(X_train, y_train)
        score_test.append(clf.score(X_test, y_test))
        score_train.append(clf.score(X_train, y_train))
    plot(range(1,22), score_test, label='test, {}'.format(estim))
    plot(range(1,22), score_train, label='train, {}'.format(estim))
    max_score = max(max_score, max(score_test))
xlabel('max depth')
ylabel('accuracy')
legend()
show()
print('maximum test score: ', max_score)

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ## Gradient boosted trees

# + {"hidden": true}
from sklearn import ensemble

# + {"hidden": true}
max_score = 0
f, ax = subplots(1, 1, figsize=(11, 6))
ax = subplot(111)
for estim in range(10,500,100):
    score_test, score_train = [], []
    for d in linspace(0.01, .7, 10):
        clf = ensemble.GradientBoostingClassifier(n_estimators= estim, learning_rate=d, subsample = .3, max_features = None, 
                                               validation_fraction=0.2,
                                               n_iter_no_change= 10, tol= .01,
                                               random_state=0)
        clf = clf.fit(X_train, y_train)
        score_test.append(clf.score(X_test, y_test))
        score_train.append(clf.score(X_train, y_train))
    plot(linspace(0.01, .5, 10), score_test, 'o', label='test, estimators: {}'.format(estim))
    plot(linspace(0.01, .5, 10), score_train, 'x', label='train, {}'.format(estim))
    max_score = max(max_score, max(score_test))
xlabel('learning rate')
ylabel('accuracy')
legend()
show()
print('maximum test score: ', max_score)

# + {"heading_collapsed": true, "cell_type": "markdown"}
# #    Nearest Neighbors

# + {"hidden": true}
from sklearn.neighbors import KNeighborsClassifier

# + {"hidden": true}
clf = KNeighborsClassifier(n_neighbors=15, weights = 'distance')
clf = clf.fit(X_train, y_train)

# + {"hidden": true}
predicted_test = clf.predict(X_test)
predicted_train = clf.predict(X_train)
print("accuracy on test data: " ,clf.score(X_test, y_test))
print(r'R^2 on test data: ', r2_score(y_test, predicted_test))

# + {"hidden": true}
score_test, score_train = [], []
for d in range(1,55):
    clf = KNeighborsClassifier(n_neighbors=d, weights = 'distance')
    clf = clf.fit(X_train, y_train)
    score_test.append(clf.score(X_test, y_test))
    score_train.append(clf.score(X_train, y_train))
f, ax = subplots(1, 1, figsize=(11, 6))
ax = subplot(111)
plot(range(1,55), score_test, label='test')
plot(range(1,55), score_train, label='training')
xlabel('no neighbors')
ylabel('accuracy')
legend()
show()
print('maximum test score: ', amax(score_test))
# -

# # Neural networks

import tensorflow as tf
import keras
from keras.utils import plot_model

# data structure for neural networks

from keras.utils import to_categorical

X_train_dnn, y_train_dnn, X_test_dnn, y_test_dnn = array(X_train), to_categorical(y_train), array(X_test), to_categorical(y_test)

# build the model

# ## "naive" DNN

model = keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(keras.layers.Dense(46, activation='relu', input_shape = X_train.shape[1:]))
model.add(keras.layers.Dropout(.2))
# Add another:
model.add(keras.layers.Dense(24, activation='relu',))
model.add(keras.layers.Dropout(.2))
# Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(9, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# show the model as a graph

# +
plot_model(model, to_file='model.png', show_shapes=True)
show('model.png')

from PIL import Image
image = Image.open("model.png")
image.show()
# -

learning = model.fit(X_train_dnn, y_train_dnn, epochs = 355, validation_split= .15)

learning.history.keys()

# summarize history for accuracy
plt.plot(learning.history['acc'])
plt.plot(learning.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(learning.history['loss'])
plt.plot(learning.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

dnn_predict = model.predict(X_test)
print(accuracy_score(argmax(dnn_predict, axis = 1), y_test))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, argmax(dnn_predict, axis = 1))
cm = cm.astype(float)/(cm.sum(axis=1)+.00001)[:, newaxis]
sns.heatmap(cm, annot = True, xticklabels=range(3,9), yticklabels=range(3,9))
xlabel('predicted'), ylabel('true')

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ## w/ bootstrapping for balanced data

# + {"hidden": true}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

num_classes = y_train.nunique()
max_class = max(y_train.value_counts())
print(max_class)
X_, y_ = [], []
for k,cl in enumerate(y_train.unique()):
    X_.append(X_train[y_train==cl].sample(max_class,replace=True).as_matrix())
    y_.append(y_train[y_train==cl].sample(max_class,replace=True).as_matrix())
X_train, y_train = array(X_).reshape(-1, X_train.shape[1]), array(y_).flatten()
print(X_train.shape)

figure()
hist(y_train)
show()

# + {"hidden": true}
X_train_dnn, y_train_dnn, X_test_dnn, y_test_dnn = array(X_train), to_categorical(y_train), array(X_test), to_categorical(y_test)

# + {"hidden": true}
learning = model.fit(X_train_dnn, y_train_dnn, epochs = 355, validation_split= .15)

# + {"hidden": true}
# summarize history for accuracy
plt.plot(learning.history['acc'])
plt.plot(learning.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(learning.history['loss'])
plt.plot(learning.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# + {"hidden": true}
dnn_predict = model.predict(X_test)
print(accuracy_score(argmax(dnn_predict, axis = 1), y_test))

# + {"hidden": true}
cm = confusion_matrix(y_test, argmax(dnn_predict, axis = 1))
cm = cm.astype(float)/(cm.sum(axis=1)+.00001)[:, newaxis]
sns.heatmap(cm, annot = True, xticklabels=range(3,9), yticklabels=range(3,9))
xlabel('predicted'), ylabel('true')

# + {"hidden": true, "cell_type": "markdown"}
# "regression to the mean"; problem with unbalanced data
# -

# ## hyperparameter optimization

import talos
from talos.model.early_stopper import early_stopper
from talos.model.normalizers import lr_normalizer

from keras.activations import relu, sigmoid, linear
from keras.optimizers import Adam

# scan parameter dictionary

p = {}
p['first_neuron'] = [8,16,32]
p['second_neuron'] = [4,8,16]
p['activations1'] = [relu, sigmoid, linear]
p['activations2'] = [relu, sigmoid, linear]
p['dropouts'] = (0., .5, 3)
p['lr'] = (0.1,10, 7) #learning rate, wrapped by lr_normalizer

p

# stratified validation set for all scans

X_train_dnn, X_val_dnn, y_train_dnn, y_val_dnn = train_test_split(X_train_dnn, y_train_dnn, test_size=0.15, random_state=42, stratify = y_train_dnn)

def optimizable_model(X_train_dnn, y_train_dnn, X_val_dnn, y_val_dnn, p):
    model = keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(keras.layers.Dense(p['first_neuron'], activation=p['activations1'], input_shape = X_train_dnn.shape[1:]))
    model.add(keras.layers.Dropout(p['dropouts']))
#     # Add another:
    model.add(keras.layers.Dense(p['second_neuron'], activation=p['activations2']))
    model.add(keras.layers.Dropout(p['dropouts']))
#     # Add a softmax layer with 10 output units:
    model.add(keras.layers.Dense(9, activation='softmax'))
    
    model.compile(optimizer=Adam(lr_normalizer(p['lr'], Adam)),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    learning = model.fit(X_train_dnn, y_train_dnn, validation_data=[X_val_dnn, y_val_dnn], epochs = 150, 
                         validation_split= .15, callbacks=early_stopper(monitor = 'val_loss', epochs = 150, patience=3, mode='moderate'), verbose = 0)
    
    return learning, model

early_stopper??

# +
# test basic functionality of network design
#optimizable_model(X_train_dnn, y_train_dnn, X_val_dnn, y_val_dnn, {"first_neuron": 2, 'activations1':relu, 'activations2':relu, 'second_neuron': 2, 'dropouts' : 0, 'lr': 1})
# -

t= talos.Scan(x = X_train_dnn, y = y_train_dnn, x_val = X_val_dnn, y_val = y_val_dnn, params = p, model = optimizable_model, dataset_name='red-wine')

t.data.sort_values('val_acc', ascending = 0).head(7)

t.details

# from saved file ("red-wine_.csv")

r = talos.Reporting("red-wine_.csv")

data = r.data
acc = pd.to_numeric(data['val_acc'])

figure()
xlabel("round")
ylabel("val_acc")
acc.plot()
show()

r.best_params(n=10)

r.correlate("val_acc"), r.correlate("acc")

# I don't think these numbers are very informative b/c I'd expect a highly non-linear (possibly non-convex) relation

r.plot_corr(['val_acc'])
tight_layout()
show()

r.plot_kde('val_acc')
show()

# does this imply a local maximum with small fluctuations around that?

# **save best model and restore**

talos.Deploy(t, 'red-wine-best-dnn')

wine_dnn = talos.Restore('red-wine-best-dnn.zip')

# test data accuracy

dnn_predict = wine_dnn.model.predict(X_test_dnn)
print(accuracy_score(argmax(dnn_predict, axis = 1), y_test))

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ### hyperas (old)

# + {"hidden": true}
def data_gen():
    from numpy import array
    wine_data = pd.read_csv('winequality-red.csv')
    X, y = wine_data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']], wine_data['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train_dnn, y_train_dnn, X_test_dnn, y_test_dnn = array(X_train), to_categorical(y_train), array(X_test), to_categorical(y_test)
    return X_train_dnn, y_train_dnn, X_test_dnn, y_test_dnn

# + {"hidden": true}
def optimizable_model(X_train_dnn, y_train_dnn,X_test_dnn, y_test_dnn):
    import numpy as np
    model = keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(keras.layers.Dense({{choice([4,8,16,32])}}, activation={{choice(['relu', 'sigmoid'])}}, input_shape = X_train_dnn.shape[1:]))
    model.add(keras.layers.Dropout({{uniform(0., .5)}}))
    # Add another:
    model.add(keras.layers.Dense(24, activation={{choice(['relu', 'sigmoid'])}}))
    model.add(keras.layers.Dropout(.2))
    # Add a softmax layer with 10 output units:
    model.add(keras.layers.Dense(9, activation='softmax'))
    
    model.compile(optimizer=tf.train.AdamOptimizer(0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    learning = model.fit(X_train_dnn, y_train_dnn,epochs = 55, validation_split= .15, verbose = 2)
    validation_acc = np.amax(learning.history['val_acc'])
    
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

# + {"hidden": true}
best_run, best_model = optim.minimize(model=optimizable_model,
                                  data= data_gen,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials(),
                                         notebook_name='wine-quality')
# -

# # "Ensembles"

# +
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load

from hyperopt import Trials, STATUS_OK, tpe, space_eval, hp, fmin
#from hyperas import optim
#from hyperas.distributions import uniform, choice
# -

# has to be run again because validation data generated in "hyperparameter optimization"
X_train_dnn, y_train_dnn, X_test_dnn, y_test_dnn = array(X_train), to_categorical(y_train), array(X_test), to_categorical(y_test)

# k-fold splitting on training data for hyperparam optimization

skf = StratifiedKFold(n_splits=6, random_state=666, shuffle=True)
splits = skf.split(X_train, y_train)

for train_index, val_index in skf.split(X_train, y_train):
   print("TRAIN:", train_index, "TEST:", val_index)
   X2_train, X2_val = array(X_train)[train_index], array(X_train)[val_index]
   y2_train, y2_val = y_train[train_index], y_train[val_index]

   X2_train_dnn, X2_val_dnn = array(X_train_dnn)[train_index], array(X_train_dnn)[val_index]
   y2_train_dnn, y2_val_dnn = y_train_dnn[train_index], y_train_dnn[val_index]

# evaluation function for RandomForest

def evaluate_forest(params):
    acc = array([])
    for train_index, val_index in skf.split(X_train, y_train):
        clf = RandomForestClassifier(**params)
#        print(array(X_train)[train_index], array(X_train)[val_index])
#        print(y_train[train_index], y_train[val_index])
        clf.fit(array(X_train)[train_index], array(y_train)[train_index])
        acc = append(acc,clf.score(array(X_train)[val_index], array(y_train)[val_index]))
    acc_mean, acc_var = mean(acc), var(acc)
    return {'loss': -acc_mean, 'loss_variance': acc_var,'status': STATUS_OK}

search_space = {'n_estimators':hp.choice('n_estimators', range(2,100,10)), 'max_depth': hp.choice('max_depth', range(1,100,10)),
            'criterion': hp.choice('criterion', ["gini", "entropy"]), 'max_features': hp.choice('max_features', range(1,11))}

trials = Trials()
best = fmin(fn=evaluate_forest,
    space= search_space,
    algo=tpe.suggest,
    max_evals=30, trials = trials)

trials.results

#print(trials.vals)
print(trials.best_trial)
print(best)

best_forest_params = space_eval(search_space, best)

# train and evaluate best random forest

best_forest = RandomForestClassifier(**best_forest_params)
best_forest.fit(X_train, y_train)

predicted_test = best_forest.predict(X_test)
print("accuracy on test data: " , best_forest.score(X_test, y_test))
print(r'R^2 on test data: ', r2_score(y_test, predicted_test))

# ## combining random forest and DNN

# I use the hyperparameter-optimized DNN from above. ideally, optimization would also be based on k-fold CV but this is too time-expensive here. 

best_forest = RandomForestClassifier(**best)
best_forest

 #model has not been saved explicitely, only parameters
best_dnn = talos.Restore('red-wine-best-dnn.zip').model

X_train_dnn, y_train_dnn, X_test_dnn, y_test_dnn = array(X_train), to_categorical(y_train), array(X_test), to_categorical(y_test)

# outputs (class predictions) of forest and DNN are inputs of ensemble network

pred_train_forest, pred_train_dnn = best_forest.predict_proba(X_train), best_dnn.predict(X_train_dnn)
pred_test_forest, pred_test_dnn = best_forest.predict_proba(X_test), best_dnn.predict(X_test_dnn)
predicted_input_train, predicted_input_test = concatenate((pred_train_forest, pred_train_dnn), axis =1), concatenate((pred_test_forest, pred_test_dnn), axis =1)

def evaluate_ensemble(params):
    p = params
    acc = array([])
    for train_index, val_index in skf.split(X_train, y_train):
        model = keras.Sequential()
        # Adds a densely-connected layer with 64 units to the model:
        model.add(keras.layers.Dense(p['first_neuron'], activation=p['activations1'], input_shape = predicted_input_test.shape[1:]))
        model.add(keras.layers.Dropout(p['dropouts']))
    #     # Add another:
        model.add(keras.layers.Dense(p['second_neuron'], activation=p['activations2']))
        model.add(keras.layers.Dropout(p['dropouts']))
    #     # Add a softmax layer with 10 output units:
        model.add(keras.layers.Dense(9, activation='softmax'))

        model.compile(optimizer=Adam(lr_normalizer(p['lr'], Adam)),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

        learning = model.fit(array(predicted_input_train)[train_index], array(y_train_dnn)[train_index], epochs = 150, validation_split= .15, callbacks=early_stopper(monitor = 'val_loss', epochs = 150, patience=3, mode='moderate'), verbose = 0)
        
        print(model.metrics_names)
        acc = append(acc,model.evaluate(array(predicted_input_train)[val_index], array(y_train_dnn)[val_index])[0])
    acc_mean, acc_var = mean(acc), var(acc)
    return {'loss': -acc_mean, 'loss_variance': acc_var,'status': STATUS_OK, 'model': model}

search_space_ensemble = {'first_neuron':hp.choice('first_neuron', [8,16,32]), 'second_neuron':hp.choice('second_neuron', [4,8,16]), 'activations1':hp.choice('activations1', [relu, sigmoid, linear]), 'activations2':hp.choice('activations2', [relu, sigmoid, linear]), 'dropouts':hp.choice('dropouts', linspace(0., .5, 3)), 'lr':hp.choice('lr', linspace(0.1,10, 7))}

# %%time
trials_ensemble = Trials()
best_ensemble = fmin(fn=evaluate_ensemble,
    space= search_space_ensemble,
    algo=tpe.suggest,
    max_evals=10, trials = trials_ensemble)

print(trials_ensemble.best_trial)
print(best_ensemble)

best_ensemble_params = space_eval(search_space_ensemble, best_ensemble)
print(best_ensemble_params)

best_ensemble_model = trials_ensemble.best_trial['result']['model']

predicted_test = best_ensemble_model.predict(predicted_input_test)
print("accuracy on test data: " , accuracy_score(argmax(predicted_test, axis = 1), y_test))

# # notes

# - unbalanced data; alternative scores
# - add info to random forests and ensembles
# - list of scores of all methods
# - stratified k-fold
# - L1 feature importance
# - hyperparameter opt (incl dropout)
# - ensemble inc neural networks
# - widgets
# - (probabilistic models?)
