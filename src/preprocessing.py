import re
import unicodedata
import emoji
import spacy
from bs4 import BeautifulSoup
from symspellpy.symspellpy import SymSpell, Verbosity
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS as stopwords

#  loads a pre-trained language processing model called 'en_core_web_sm' from the SpaCy library for English language tasks.
# Load SpaCy model 
nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()
stemmer = PorterStemmer()

# Load symspell dictionary
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Contraction mapping
# Contraction to Expansion
# Contraction to Expansion in NLP refers to transforming contracted words (shortened forms like "I'm" for "I am") into their fully expanded forms ("I am"). This is done to standardize and normalize text for analysis, as well as to ensure consistency in language representation.
 
contraction = {
"a'ight" :	"alright",
"ain't" : "am not",
"amn't" :	"am not",
"aren't" :	"are not",
"‘bout" :	"about",
"can't"	: "cannot",
"cap’n" :	"captain",
"'cause" :	"because",
"’cept" :	"except",
"could've":	"could have",
"couldn't"	: "could not",
"couldn't've"	: "could not have",
"dammit" :	"damn it",
"daren't"	: "dare not ",
"daresn't":	"dare not",
"dasn't":	"dare not",
"didn't" :	"did not",
"don't" : "do not",
"he 'll" : "he will ",
"hadn't": "had not",
"hadn't": "had not have",
"has't" : "has not",
"have't": "have not",
"he'd": "he would",
"he'll" : "he will ",
"he'll've" : "he will	have",
"he 's" : "he is",
"how 'd " : "how d id",
"how'd'y " : "how do you ",
"how'll" :	"how will" ,
"how's" : "f how does" ,
"i'd" : "i would ",
    "i'd've" : "i would	have" ,
    "i'll" : "i will" ,
    "i'll've" : "i will	have" ,
    "i'm" : "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd " :	"it would ",
    "it'd've " : "it wou ld have" ,
    "it'll" : "it will ",
    "it'll ' ve" :	"it will have" ,
    "it's" : "it is",
    "let's" : "let us" ,
    "ma'am" : "madam ",
    "mayn't " : "may not ",
    "might've" : "might	have" ,
    "mightn't " : "might	not",
    "must've" : "must have" ,
    "mustn't " : "must not ",
    "mustn't've" : "mu st not	have" ,
    "needn't " : "need not ",
    "needn't ' ve" : "need	not	have" ,
    "o'clock" : "of	the clock ",
    "oughtn't " : "ought	not ",
    "oughtn't ' ve" : "ought not have" ,
    "shan't" : "shall not ",
    "sha'n't " : "shall not ",
    "shan't've" :	"sha ll not",
    "she'd" :	"she would ",
    "she'd've " : "she would",
    "she'll" : "she will ",
    "she'll've" : "she will",
    "she's" : "she is",
    "should've" : "should	have",
    "shouldn't ": "should	not",
    "shouldn't've" : "should	not	have" ,
    "so've" : "so have" ,
    "so's":"so  is",
    "that'd" :	"that	wou ld",
    "that'd ' ve" :	"that wou ld	have",
    "that's" : "that	is",
    "there'd" : "there would",
    "there'd've" : "the re wou ld	have",
    "there's" : "there	is",
    "they'd" : "they would",
    "they'd've" :	"they wou ld	have",
    "they'll" : "they will",
    "they'll've" : "they will have" ,
    "they're" : "t hey are" ,
    "they've" : "they have" ,
    "to've" : "to have"  ,
    "wasn't ": " was not ",
    "we've": "we have"
}

def char_counts(x):
    return len(''.join(x.split()))

# create a function to for contraction words
def cont_to_exp(x):
    if type(x) is str:
        for key in contraction:
            value = contraction[key]
            x = x.replace(key, value)
        return x
    else:
        return x
# removing accented characters
def remove_accented_chars(x):
  x = unicodedata.normalize('NFKD',x).encode('ascii','ignore').decode('utf-8','ignore')
  return x

def make_to_base(x):
    x = str(x)
    x_list = []
    doc = nlp(x)

    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
           lemma = token.text

        x_list.append(lemma)
    return ' '.join(x_list)

def correct_symspell(text):
    corrected_words = []
    for word in str(text).split():
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)

# Negation handling (not good  not_good) 
def handle_negations(text):
    return re.sub(r"\bnot\s+(\w+)", r"not_\1", text)

# keep only meaningful parts of speech
def filter_pos(text):
    doc = nlp(text)
    allowed_pos = ['ADJ', 'NOUN', 'VERB', 'ADV']
    return ' '.join([token.text for token in doc if token.pos_ in allowed_pos])
 
# Remove convert emoji into text
def convert_emojis_to_text(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def stemming(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

def polarity(text):
    return TextBlob(text).sentiment.polarity

def sentiment_label(polarity_score):
    if polarity_score < 0:
        return "Negative"
    elif polarity_score == 0:
        return "Neutral"
    else:
        return "Positive"

def clean_text(text):
    # changing all the letters in text to lowercase  This is done to ensure uniformity and consistency in the text data.

    text = str(text).lower()
    text = cont_to_exp(text)
    text = re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', '', text)
    #  remove URL
    text = re.sub(r'(http|https|ftp|ssh)://\S+', '', text)
    # Remove re-tweets
    text = re.sub(r'\brt\b', '', text)
    # Remove special chars
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove multiple space
    text = ' '.join(text.split())
# Remove HTML tages using beautiful soup
    text = BeautifulSoup(text, 'lxml').get_text()
    text = remove_accented_chars(text)
    text = make_to_base(text)
    text = correct_symspell(text)
    text = handle_negations(text)
    text = filter_pos(text)
    text = convert_emojis_to_text(text)
    text = stemming(text)
    return text
