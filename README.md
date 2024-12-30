# LLM GreenWashing Detection

This repository contains the code for the LLM GreenWashing Detection project.

## Quick Start

### Install Dependencies

We recommand using Python 3.10 or above. You can install the dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### Pipeline for TF-IDF and Keyword Scoring

**1. Auto-extract symbolic and exact keywords from the ESG reports**

Run the following command to extract symbolic and exact keywords from the ESG reports, where:
- ``--data_path`` is the path to the ESG reports (`.txt` files).
- ``--api_key`` is the API key for the DeepSeek API.
-  ``--outdir`` is the path to the output directory.
-  ``--num_reports`` is the number of reports to be examined.

**Example:**

```bash
python keyword_extraction.py --data_path ./data --api_key <your_api_key> --outdir ./jieba_wordlist --num_reports 10
```

**2. Train the scorers**

Train the scorers by running the following command, where:
- ``--data_path`` is the path to the ESG reports.
- ``--scoring_method`` is the scoring methods to use, separated by commas.
- ``--save_path`` is the path to save the trained scorers.

**Example:**

```bash
python train_scorers.py --data_path ./data --scoring_method kw,tfidf --save_path ./pretrained_scorer
```

**3. Score the ESG reports**

Score the ESG reports by running the following command, where:
- ``--data_path`` is the path to the ESG reports.
- ``--scoring_method`` is the scoring methods to use, separated by commas.
- ``--outdir`` is the path to save the scoring results.
- ``--pretrained_path`` is the path to the pretrained scorers.

**Example:**

```bash
python main_scoring.py --data_path ./data --scoring_method kw,tfidf --outdir ./scoring_results --pretrained_path ./pretrained_scorer
```
