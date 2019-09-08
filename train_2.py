import pandas as pd
import numpy as np
data = pd.read_csv("training_data_DS_Specialist.csv")
data["raw_labels"] = data["labels"].apply(lambda x: x)
data["labels"] = data["labels"].apply(lambda x: eval(x))
all_categories = data["labels"].apply(lambda x: '_'.join(x))
max_heirarchy = np.max(data["labels"].apply(lambda x: len(x)).values)
data['raw_labels'] = data.labels.apply(lambda x : ' > '.join(x))
data.head()

from collections import Counter
# default_depth = max_heirarchy
default_depth = 2
min_samples = 50

def depth(field, n, sep=' > '):
    if n <= 0:
        return field
    return sep.join(field.split(sep, n)[: n])

categories = Counter(
    depth(x, default_depth)
    for x in structured_df.raw_labels.values.tolist()
)


categories_filter = {}
for x in categories:
    if any([x.startswith(j) for j in domains]) or not domains:
        if categories[x] > min_samples:
            categories_filter[x] = categories[x]


categories_dict = {}

for cat in sorted(categories_filter):
    # noinspection PyRedeclaration
    parent = categories_dict

    for i in cat.split(' > '):
        parent = parent.setdefault(i, {})


def pretty(d, indent=0):
    for key, value in d.items():
        print(u'{} {} ({})'.format('    ' * indent, key, len(value)))
        pretty(value, indent + 1)


class_hierarchy = {}
class_hierarchy['ROOT'] = list(categories_dict)

def t(d):
    for k in d.keys():
        class_hierarchy[k] = list(d[k])
        t(d[k])
t(categories_dict)
#pretty(categories_dict)



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


models = {}
feature_selectors = {}
for k in class_hierarchy.keys():
    print(k)
    v = class_hierarchy[k]
    X_all = []
    y_all = []
    for classs in v:
        print("Processing",classs)
        temp_df = train_df[train_df['raw_labels'].str.contains(classs)][['titles']]
        temp_df['features'] = temp_df['titles'].apply(lambda x: featurizer.transform([x]).toarray()[0])

        X_all.extend(temp_df['features'].values)
        y_all.extend([classs] * temp_df.shape[0])

        assert len(X_all) == len(y_all), "Error, dimension mismatch"
        print("Adding ", temp_df.shape[0],  " records")

        print("Total Records updated to : ", len(X_all))

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








