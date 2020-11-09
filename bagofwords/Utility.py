import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


class Utility(object):

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=False ):
        # Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #
        # Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # Remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # Return a list of words
        return(words)

    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        # 1. Split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                sentences.append(Utility.review_to_wordlist( raw_sentence, remove_stopwords))
        # Return the list of sentences
        return sentences
