import codecs
import gensim
stopwords = codecs.open("stopwords.txt", "r", "utf-8")
stopwords = stopwords.readlines()
stopwords = stopwords[0].split(",")
dictionary = gensim.corpora.Dictionary([["hei", "an"], ["test", "hade"]])
id = dictionary.token2id["an"]


print(dictionary)