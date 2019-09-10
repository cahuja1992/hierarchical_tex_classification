import pandas as pd
import numpy as np
data = pd.read_csv("data/training_data_DS_Specialist.csv")
data["raw_labels"] = data["labels"].apply(lambda x: x)
data["labels"] = data["labels"].apply(lambda x: eval(x))
all_categories = data["labels"].apply(lambda x: '_'.join(x))
max_heirarchy = np.max(data["labels"].apply(lambda x: len(x)).values)
data['raw_labels'] = data.labels.apply(lambda x : ' > '.join(x))
data.head()

import pickle
class_hierarchy = pickle.load(open("class_hierarchy", "rb"))


import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import wordpunct_tokenize
from nltk.corpus import wordnet as wn
from functools import lru_cache
from nltk.tag.perceptron import PerceptronTagger
import matplotlib.pyplot as plt


# Initiate lemmatizer
wnl = WordNetLemmatizer()

# Load tagger pickle
tagger = PerceptronTagger()

# Lookup if tag is noun, verb, adverb or an adjective
tags = {'N': wn.NOUN, 'V': wn.VERB, 'R': wn.ADV, 'J': wn.ADJ}

# Memoization of POS tagging and Lemmatizer
lemmatize_mem = lru_cache(maxsize=10000)(wnl.lemmatize)
tagger_mem = lru_cache(maxsize=10000)(tagger.tag)


def tokenizer(text):
    for token in wordpunct_tokenize(text):
        if token not in ENGLISH_STOP_WORDS:
            tag = tagger_mem(frozenset({token}))
            yield lemmatize_mem(token, tags.get(tag[0][1],  wn.NOUN))



import pickle
featurizer = pickle.load(open("terms_vectorizer", "rb"))

train_df = data

import tensorflow as tf
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


models = {}
feature_selectors = {}
metadata = {}
print(class_hierarchy)
from tqdm import tqdm
for k in class_hierarchy.keys():
    print(k)
    writer = tf.python_io.TFRecordWriter(f"tfrecord/{k}_train.tfrecords")
    v = class_hierarchy[k]
    X_all = []
    y_all = []
    metadata[k] = {}
    for idx, classs in enumerate(v):
        print("Processing",classs)
        metadata[k][idx] = classs
        temp_df = train_df[train_df['raw_labels'].str.contains(classs)][['titles']]
        for i, r in tqdm(temp_df.iterrows()):
            x = featurizer.transform([r['titles']]).toarray()[0].tostring()
            feature = { 'label': _int64_feature(idx),
                        'title': _bytes_feature(x) }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    writer.close()
    print(metadata)
with open("metadata.json", "w") as f:
    json.dump(metadata, f)

        #temp_df['features'] = temp_df['titles'].apply(lambda x: featurizer.transform([x]).toarray()[0])
        
        #X_all.extend(temp_df['features'].values)
        #y_all.extend([classs] * temp_df.shape[0])

        #assert len(X_all) == len(y_all), "Error, dimension mismatch"
        #print("Adding ", temp_df.shape[0],  " records")

        #print("Total Records updated to : ", len(X_all))
"""
    if(len(X_all) > 50) and np.unique(y_all).shape[0] > 1:
        print("Fitting the model")
        from sklearn.utils import class_weight
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        from sklearn.feature_selection import SelectKBest, chi2
        assert len(X_all) == len(y_all), "Dimension of X and Y not same"


        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_all), y_all)
        class_weights= dict(list(zip(np.unique(y_all), class_weights)))
        for n in np.arange(100,9999,100):
            ch2 = SelectKBest(chi2, k=n)
            X_selected = ch2.fit_transform(X_all, y_all)
            feature_selectors[k] = ch2

    #    Too Slow on the laptop and colab
    #     classifier = SVC()
    #     grid_param = {'C' : [0.01, 0.1, 1, 10],
    #                   'kernel': ('rbf', 'linear'),
    #                   'class_weight': [class_weights],
    #                  'probability'=[True]}
        classifier = SGDClassifier()

        grid_param = {
            'loss':['log'],
            'class_weight': [class_weights],
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'n_jobs': [-1]
        }
        gd_sr = GridSearchCV(estimator=classifier,
                         param_grid=grid_param,
                         scoring='accuracy',
                         cv=5,
                         n_jobs=-1)
        gd_sr.fit(X_selected, y_all)

        models[k] = gd_sr.best_estimator_
        print("Completed")







def predict_hierarchy(title):
    def predict_by_key(k):

        if k in feature_selectors.keys():
            f = feature_selectors[k].transform(featurizer.transform([title]).toarray())
            label = models[k].predict(f)
            probability = models[k].predict_proba(f)
            return label[0], np.max(probability)

    final_label = []
    label_probabs = []

    def pred(key):
        if predict_by_key(key):
            if len(class_hierarchy[key]) > 0:
                label, conf = predict_by_key(key)
            else:
                label, conf = key, 1

            final_label.append(label)
            label_probabs.append(str(conf))
            pred(label)


    pred('ROOT')
    return ' > '.join(final_label), ' > '.join(label_probabs)






"""

