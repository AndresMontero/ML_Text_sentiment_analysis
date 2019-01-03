# This python file contains functions for preprocessing raw datasets

import re
import nltk

from textblob import TextBlob

contractions_dict = {
    """
        DESCRIPTION:
        Dictionary used for expanding contractions
    """
    "didnt": "did not", "don\'t": "do not",
    "didn\'t": "did not", "don\'t": "do not",
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they shall have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

def one_space(tweet):
    """
    DESCRIPTION: Removes extra spaces between words, ensures only one space is left
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            modified tweet, containing only one space between words
            (e.g. "today is     friday" outputs "today is friday")
    """
    tweet = re.sub("\s\s+", " ", tweet)
    return tweet

def replace_moreletters(tweet):
    """
    DESCRIPTION: Replaces by 1 repeated letters when there are more than two repeated letters
    INPUT:
            tweet:  a string
    OUTPUT:
            A tweet without letters repeating more than two times
            (e.g. "I am haaaaaapy" outputs "I am haapy")
    """

    pattern = re.compile(r"(.)\1{3,}", re.DOTALL)
    return pattern.sub(r"\1\1", tweet + "rep*")

def interpret_emoji(tweet):
    """
    DESCRIPTION: 
                transforms emoticons to sentiment tags e.g :) --> <smile>
    INPUT: 
            tweet: a tweet as a python string
    OUTPUT: 
            transformed tweet as a python string
            (e.g. "today is friday :-) <3" outputs "today is friday <smile> <heart>")
    """
    # Construct emojis
    hearts = ["<3", "♥"]
    eyes = ["8", ":", "=", ";"]
    nose = ["'", "`", "-", r"\\"]
    smilefaces = []
    sadfaces = []

    for e in eyes:
        for n in nose:
            for s in ["\)", "d", "]", "}", ")"]:
                smilefaces.append(e + n + s)
                smilefaces.append(e + s)
            for s in ["\(", "\[", "{", "(", "["]:
                sadfaces.append(e + n + s)
                sadfaces.append(e + s)
            # reversed
            for s in ["\(", "\[", "{", "[", "("]:
                smilefaces.append(s + n + e)
                smilefaces.append(s + e)
            for s in ["\)", "\]", "}", ")", "]"]:
                sadfaces.append(s + n + e)
                sadfaces.append(s + e)
         
    smilefaces = set(smilefaces)
    sadfaces = set(sadfaces)
    t = []
    for w in tweet.split():
        if (w in hearts):
            t.append("<heart>")
        elif (w in smilefaces):
            t.append("<smile>")
        elif (w in sadfaces):
            t.append("<sad>")
        else:
            t.append(w)
    return (" ".join(t)).strip()


##Code modified from https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
def expand_contractions(s, contractions_dict=contractions_dict):
    """
    DESCRIPTION: Expands contractions of all words (contracted) in the tweet
    INPUT:
            tweet: a tweet as a python string
    OUTPUT:
            modified tweet, replacing contracted words by their expanded version
            (e.g. "don't do that, today isn't friday" outputs "do not do that, today is not friday")
    """
    
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)









