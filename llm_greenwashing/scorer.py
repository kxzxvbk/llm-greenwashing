import os
import math
from tqdm import tqdm

import jieba
jieba.load_userdict('jieba_wordlist/financial_wordlist.txt')
jieba.load_userdict('jieba_wordlist/exact_keywords.txt')
jieba.load_userdict('jieba_wordlist/symbolic_keywords.txt')


def load_keywords():
    # Load exact keywords
    with open("jieba_wordlist/exact_keywords.txt", encoding='utf-8') as f:
        pos_words = f.read().strip().split()

    # Load symbolic keywords
    with open("jieba_wordlist/symbolic_keywords.txt", encoding='utf-8') as f:
        neg_words = f.read().strip().split()
    return pos_words, neg_words


class KeywordScorer:
    def __init__(self):
        # Load some pre-defined exact and symbolic keywords.
        self.pos_keywords, self.neg_keywords = load_keywords()

    def train(self, corpus):
        return

    def score(self, total_txt):
        # Remove all whitespace characters from text
        cleaned_text = ''.join(total_txt.split())

        # Calculate the number of positive words and negative words respectively.
        score = 0

        # The larger the score, the more likely the text is greenwashing.
        for kw in self.pos_keywords:
            score -= cleaned_text.count(kw)
        for kw in self.neg_keywords:
            score += cleaned_text.count(kw)

        # Return the sign of final score.
        if score > 0:
            return 1
        elif score == 0:
            return 0
        else:
            return -1


class TFIDFScorer:
    def __init__(self):
        # Load some pre-defined exact and symbolic keywords.
        pos_keywords, neg_keywords = load_keywords()

        # Store the ``dfc`` value for each keyword
        self.pos_dfc = {k: 0 for k in pos_keywords}
        self.neg_dfc = {k: 0 for k in neg_keywords}

        # Stepup stop words
        with open('jieba_wordlist/stop_words.txt', encoding='utf-8') as f:
            self.stop_words = f.read().strip().split()

        # Store the ``n`` value
        self.n = 0

    def train(self, corpus):
        # Iterate through the total corpus to calculate ``dfc`` for each pos-word and neg-word
        print('Begin to train ...')
        files = os.listdir(corpus)

        # ``n`` is set to the total number of documents
        self.n = len(files)

        # Begin to calculate ``dfc`` value for each keyword
        for i in tqdm(range(len(files))):
            file = files[i]
            # Filter files with wrong format
            if not file.endswith('.txt'):
                continue

            # Load the content for each document.
            with open(os.path.join(corpus, file), encoding='utf-8') as f:
                text = f.read().strip()

            # Cut the document into words
            words = jieba.lcut(text)

            # If a keyword is in the words, its ``dfc+=1``
            for k in self.pos_dfc:
                if k in words:
                    self.pos_dfc[k] += 1
            for k in self.neg_dfc:
                if k in words:
                    self.neg_dfc[k] += 1

    def score(self, total_txt):
        # Cut the document into words
        words = jieba.lcut(total_txt)

        # Calculate the number of words after removing stop words.
        word_len = 0
        for word in words:
            if word not in self.stop_words:
                word_len += 1

        score = 0
        # Calculate the weights for positive words and negative words respectively and sum them up together.
        # Dealing with positive words.
        for k, dfc in self.pos_dfc.items():
            tfci = words.count(k)
            if tfci > 0:
                score -= (1 + math.log(tfci)) / (1 + math.log(word_len)) * math.log(self.n / dfc)

        # Dealing with negative words.
        for k, dfc in self.neg_dfc.items():
            tfci = words.count(k)
            if tfci > 0:
                score += (1 + math.log(tfci)) / (1 + math.log(word_len)) * math.log(self.n / dfc)

        return score
