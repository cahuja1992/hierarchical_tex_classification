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
from sklearn import svm

with open("products.json") as f:
    products = json.load(f)

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

# Pipeline definition
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        tokenizer=tokenizer,
        ngram_range=(1, 2),
        stop_words=ENGLISH_STOP_WORDS,
        sublinear_tf=True,
        min_df=0.00009
    )),
    ('classifier',SGDClassifier(alpha=1e-4, n_jobs=-1)
        
    )
])

y_pred = cross_val_predict(
    pipeline, products.get('data'),
    y=products.get('target'),
    cv=10
)


cr = classification_report(
    products.get('target'), y_pred,
    target_names=products.get('target_names'),
    digits=3
)

label_length = len(
    sorted(products['target_names'], key=len, reverse=True)[0]
)
label_length

short_labels = []
for i in products['target_names']:
    short_labels.append(
        ' '.join(map(lambda x: x[:3].strip(), i.split(' > ')))
    )

toBePersisted = dict({
    'model': pipeline,
    'metadata': {
        'name': 'filtered_all_by_minexamples_50_and_depth_10',
        'author': 'Chirag Ahuja',
        'metrics': {
            'classification_report': cr
        }
    }
})

from joblib import dump
dump(toBePersisted, 'filtered_all_by_minexamples_50_and_depth_10.joblib')

# Printing Classification Report
print('{label:>{length}}'.format(
    label='Classification Report',
    length=label_length
), cr, sep='\n')

# Pretty printing confusion matrix
print('{label:>{length}}\n'.format(
    label='Confusion Matrix',
    length=abs(label_length - 50)
))
for index, val in enumerate(cm):
    print(
        '{label:>{length}} {prediction}'.format(
            length=abs(label_length - 50),
            label=short_labels[index],
            prediction=''.join(map(lambda x: '{:>5}'.format(x), val))
        )
    )    
   
