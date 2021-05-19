from nltk.util import everygrams
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def grams(text):
    #Character grams
    for i in list(everygrams(''.join([c for c in text if c is not ' ']), min_len=1, max_len=4)):
        yield i
    #Word grams
    for i in list(everygrams(text.split(' '), min_len=1, max_len=1)):
        yield i
