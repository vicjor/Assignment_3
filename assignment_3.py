import random; random.seed(123)
import codecs
import string
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
import gensim
from nltk.tokenize import word_tokenize
# Dokumentasjon på NLTK: https://kite.com/python/docs/nltk.FreqDist

## OPPGAVE 1 ##

f = codecs.open("pg3300.txt", "r", "utf-8")
file = f.readlines()
all_text = ""

# Lager en lang streng med all texten
for line in file:
    all_text += line


# Deler opp teksten der det er to linjeskift etter hverandre
paragraphs = all_text.split("\r\n\r\n")

number_of_paragraphs = len(paragraphs)

filtered_gutenberg = []

# Fjerner alle avsnitt med Gutenberg i seg. Lagrer disse i en ny liste.
for par in paragraphs:
    if "Gutenberg" not in par:
        filtered_gutenberg.append(par)

# Gjør om slik at hvert element i listen er en liste med ord
paragraphs_listed_words = []
for par in filtered_gutenberg:
    paragraphs_listed_words.append(par.split())


# Fjerner alle tegn og gjør om til små bokstaver. Gjør flere deloppgaver i samme løkke for å slippe mange doble løkker og lang kjøretid.
par_counter = 0
word_counter = 0

stemmer = PorterStemmer()
freqDist =  FreqDist()

for par in paragraphs_listed_words:
    word_counter = 0
    for word in par:
        paragraphs_listed_words[par_counter][word_counter] = stemmer.stem(word.strip(string.punctuation + "\n\r\t").lower())
        # Teller opp antall forekomster av hver ord
        freqDist[paragraphs_listed_words[par_counter][word_counter]] += 1
        word_counter += 1
    # Fjerner alle tomme strenger
    paragraphs_listed_words[par_counter] = list(filter(None, paragraphs_listed_words[par_counter]))
    par_counter += 1

# Fjerner til slutt alle tomme lister
paragraphs_listed_words = list(filter(None, paragraphs_listed_words))



## OPPGAVE 2 ##

dictionary = gensim.corpora.Dictionary(paragraphs_listed_words)
print(dictionary)

stopwords = codecs.open("stopwords.txt", "r", "utf-8")
stopwords = stopwords.readlines()
stopwords = stopwords[0].split(",")
# Stopwords er nå en liste med alle ordene
stop_ids = []
#
for word in stopwords:
    print(word)
    id = dictionary.token2id[word]
    stop_ids.append(id)

dictionary.filter_tokens(stop_ids)

