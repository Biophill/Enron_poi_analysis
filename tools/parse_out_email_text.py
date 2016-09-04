#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
		
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
    """

    f.seek(0)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words
        word_list = text_string.split()
        stemmer = SnowballStemmer("english")
        stemmed_list = map(stemmer.stem, word_list)
        stemmed_string = " ".join(stemmed_list)


    return stemmed_string

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()

