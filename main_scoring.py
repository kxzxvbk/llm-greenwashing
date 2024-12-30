import os
import pickle
import pandas as pd
from tqdm import tqdm
import argparse

from llm_greenwashing.utils import post_process


def init_scorers(pretrained_path: str, scoring_method: list[str]):
    """
    Overview:
        Initialize the scorers and load the pretrained weights.
    Arguments:
        - pretrained_path: The path to the pretrained weights.
        - scoring_method: The scoring methods to use.
    Returns:
        - scorers: A dictionary of scorers.
    """
    scorers = {}
    for method in scoring_method:
        if method == 'kw':
            if os.path.exists(os.path.join(pretrained_path, 'kw_scorer.pkl')):
                with open(os.path.join(pretrained_path, 'kw_scorer.pkl'), 'rb') as f:
                    kw_scorer = pickle.load(f)
            else:
                raise ValueError(f'Pretrained kw_scorer not found in {pretrained_path}')
            scorers['kw'] = kw_scorer
        elif method == 'tfidf':
            if os.path.exists(os.path.join(pretrained_path, 'tfidf_scorer.pkl')):
                with open(os.path.join(pretrained_path, 'tfidf_scorer.pkl'), 'rb') as f:
                    tfidf_scorer = pickle.load(f)
            else:
                raise ValueError(f'Pretrained tfidf_scorer not found in {pretrained_path}')
            scorers['tfidf'] = tfidf_scorer
        else:
            raise ValueError(f'Unknown scoring method: {method}')
    return scorers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data', required=False, help='The path to the corpus.')
    parser.add_argument('--outdir', default='./esg_evaluation_results.csv', required=False, help='The path to the output file.')
    parser.add_argument('--scoring_method', default='kw, tfidf', required=False, help='The scoring methods to use, separated by commas.')
    parser.add_argument('--pretrained_path', default='./pretrained_scorer', required=False, help='The path to the pretrained scorers.')
    args = parser.parse_args()

    corpus_path = args.data_path
    scoring_method = map(str.strip, args.scoring_method.split(','))
    scorers = init_scorers(args.pretrained_path, scoring_method)

    # Store the score results.
    res = {'file_name': []}
    res.update({method + '_score': [] for method in scoring_method})

    # Predict the score for each document.
    print('Scoring for each documents ...')
    for file in tqdm(os.listdir(corpus_path)):
        # Filter files with wrong format.
        if not file.endswith('.txt'):
            continue

        # Load the document content.
        with open(os.path.join(corpus_path, file), encoding='utf-8') as f:
            document_text = f.read().strip()

        # Calculate the scores using different scorer and save them.
        res['file_name'].append(file)
        for method in scoring_method:
            res[method + '_score'].append(scorers[method].score(document_text))

    res = post_process(pd.DataFrame(res))
    res.to_csv(os.path.join(args.outdir, 'esg_evaluation_results.csv'))
