import pandas as pd
import numpy as np
data = pd.read_csv("data/training_data_DS_Specialist.csv")
data["raw_labels"] = data["labels"].apply(lambda x: x)
data["labels"] = data["labels"].apply(lambda x: eval(x))

cat_1 = data["labels"].apply(lambda x: x[0])
cat_1_unique = np.unique(cat_1, return_counts=True)
print(cat_1_unique)
all_categories = data["labels"].apply(lambda x: '_'.join(x))
max_heirarchy = np.max(data["labels"].apply(lambda x: len(x)).values)


domains = [ v for id, v in enumerate(cat_1_unique[0]) if cat_1_unique[1][id] >= 50]
print(domains)


data['raw_labels'] = data.labels.apply(lambda x : ' > '.join(x))
data.head()


# default_depth = max_heirarchy
default_depth = 2
min_samples = 50


from collections import Counter

def depth(field, n, sep=' > '):
    if n <= 0:
        return field
    return sep.join(field.split(sep, n)[: n])

categories = Counter(
    depth(x, default_depth)
    for x in data.raw_labels.values.tolist()
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
pretty(categories_dict)





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



featurizer = Pipeline([
    ('vectorizer', TfidfVectorizer(
        tokenizer=tokenizer,
        ngram_range=(1, 2),
        stop_words=ENGLISH_STOP_WORDS,
        sublinear_tf=True,
        min_df=0.00009
    ))  
])
features = featurizer.fit_transform(data['titles'])
print(features.shape)
dense_features = features.toarray()


import pickle
pickle.dump(featurizer, open("terms_vectorizer", "wb"))
pickle.dump(class_hierarchy, open("class_hierarchy", "wb"))
"""
products = {
    'target_names': list(categories_filter),
    'target': [],
    'target_leaf': [],
    'data': []
}
for _, product in structured_df.iterrows():
    if depth(product['raw_labels'], default_depth) in products.get('target_names'):
        if categories.get(depth(product['raw_labels'], default_depth)):
            products['target'].append(
              products.get('target_names').index(depth(product['raw_labels'], default_depth))
            )
            products['target_leaf'].append(
              depth(product['raw_labels'], default_depth).split(' > ')[-1]
            )
            products['data'].append(u'{}.'.format(product['titles'] or ''))

with open("products.json", "w") as f:
    json.dump(products, f)
"""

