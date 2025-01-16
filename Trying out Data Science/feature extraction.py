#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:39:32 2024

@author: dex
"""
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample data
texts = ["Machine learning is fun", "Learning new features is great"]

# Bag of Words
bow = CountVectorizer()
bow_features = bow.fit_transform(texts)
print("Bag of Words Features:\n", bow_features.toarray())
print("Vocabulary:\n", bow.vocabulary_)

tfidf = TfidfVectorizer()
tfidf_features = tfidf.fit_transform(texts)
print("\nTF-IDF Features:\n", tfidf_features.toarray())

