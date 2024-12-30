import os
import pickle
import argparse
from typing import List

from llm_greenwashing.scorer import KeywordScorer, TFIDFScorer


def init_scorers(scoring_method: List[str]):
    """
    Overview:
        Initialize the scorers.
    Arguments:
        - scoring_method: The scoring methods to use.
    Returns:
        - scorers: A dictionary of scorers.
    """
    scorers = {}
    for method in scoring_method:
        if method == 'kw':
            kw_scorer = KeywordScorer()
            scorers['kw'] = kw_scorer
        elif method == 'tfidf':
            tfidf_scorer = TFIDFScorer()
            scorers['tfidf'] = tfidf_scorer
        else:
            raise ValueError(f'Unknown scoring method: {method}')
    return scorers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data', required=False, help='The path to the corpus.')
    parser.add_argument('--scoring_method', default='kw, tfidf', required=False, help='The scoring methods to use, separated by commas.')
    parser.add_argument('--save_path', default='./pretrained_scorer', required=False, help='The path to save the trained scorers.')
    args = parser.parse_args()

    corpus_path = args.data_path
    scoring_method = map(str.strip, args.scoring_method.split(','))

    scorers = init_scorers(scoring_method)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    for method, scorer in scorers.items():
        print(f'Training {method} scorer ...')
        scorer.train(corpus_path)
        with open(os.path.join(args.save_path, f'{method}_scorer.pkl'), 'wb') as f:
            pickle.dump(scorer, f)
