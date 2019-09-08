# Product Categorization

## Preprocessing
1. **Text Vectorization:** tfidf and word2vec
2. **Pre-processing:** n-grams, removing stop-words, stemming/lemmetisation etc. n-grams takes blocks of 2 consecutive words or 3 consecutive words, in-addition to single words, while creating a TDM. Removing the stop words excludes the punctuation marks and other inconsequential words like articles (a, an, the) prepositions and conjunctions (about, among, anyhow etc.) from the TDM. Stemming and lemmetisation prune a word to its root. Plural becomes singular, different tense variants reduce to their simple present form.

```python
python3 data_prep.py
```

## Method 1(All the levels together as one category):
In this approach, the whole block shown in the example above is treated as one category. If there are 14 distinct categories across products in level 1, 15 in level 2 and 20 in level 3, there would be 3000(10*15*20) such categories. The number of categories would increase exponentially with number of distinct categories in each level. A large number of categories is difficult to handle and leads to lower accuracies in the text classification methods because of a lot of data imbalance and too many classes.
```python
python3 train_1.py
```

## Method 2(Different classification for different levels):
This method entails a nested/iterative approach. In the first pass, level 1 of hierarchy is predicted. In the 2nd pass, a separate model is run for each category in level 1 to predict level 2 category. This method gives better accuracy but increases the number of models which needs to be trained.
```python
python3 train_2.py
```
