import codecs
import gensim
import string
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
stopwords = codecs.open("stopwords.txt", "r", "utf-8")
stopwords = stopwords.readlines()
stopwords = stopwords[0].split(",")
dictionary = gensim.corpora.Dictionary([["hei", "an"], ["test", "hade"]])
id = dictionary.token2id["an"]




f = codecs.open("pg3300.txt", "r", "utf-8")
file = f.readlines()
all_text = ""
for line in file:
    all_text += line

print(all_text[0:500])