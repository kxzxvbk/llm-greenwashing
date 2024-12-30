import os
import argparse
import random
from tqdm import tqdm

from prompt_templates import keyword_extraction_template
from llm_greenwashing.llm_utils import fill_in_template, DeepseekAPIClient, extract_json


def split_content(content, max_length=1024, max_chunk=10):
    # Split the content into chunks of max_length.
    start = 0
    res = []
    while start < len(content):
        end = min(len(content), start + max_length)
        res.append(content[start: end])
        start = end
    if len(res) > max_chunk:
        res = random.sample(res, max_chunk)
    return res


def preload_keywords():
    # Load the pre-extracted keywords.
    with open('./jieba_wordlist/orig_symbolic_keywords.txt', 'r', encoding='utf-8') as f:
        symbolic_keywords = set(f.read().splitlines())
    with open('./jieba_wordlist/orig_exact_keywords.txt', 'r', encoding='utf-8') as f:
        exact_keywords = set(f.read().splitlines())
    return symbolic_keywords, exact_keywords


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data', required=False, help='The path to the corpus.')
    parser.add_argument('--api-key', required=True, help='The API key for the DeepSeek API.')
    parser.add_argument('--outdir', default='./jieba_wordlist', required=False, help='The path to save the extracted keywords.')
    parser.add_argument('--num_reports', default=10, required=False, help='The number of reports to extract keywords from.')
    args = parser.parse_args()
    root_path = args.data_path

    api_agent = DeepseekAPIClient(api_key=args.api_key)
    symbolic_keywords, exact_keywords = preload_keywords()

    target_files = os.listdir(root_path)
    target_files = random.sample(target_files, args.num_reports)
    
    # Extract keywords from the target files.
    for i in tqdm(range(len(target_files))):
        file = target_files[i]
        file_path = os.path.join(root_path, file)
        with open(file_path, encoding='utf-8') as f:
            file_content = f.read()
        chunked_content = split_content(file_content)
        for chunk in chunked_content:
            try:
                prompt = fill_in_template(keyword_extraction_template, document=chunk)
                query = [{"role": "user", "content": prompt}]
                response = api_agent.generate(query)
                response = eval(extract_json(response))
                symbolic_keywords |= set(response['象征性环境行动关键词'])
                exact_keywords |= set(response['实际性环境行动关键词'])
            except Exception as e:
                print(f'Error in file {file}: {e}')
                pass
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    # Save the extracted keywords.
    with open(os.path.join(args.outdir, "symbolic_keywords.txt"), "w", encoding="utf-8") as f:
        for word in symbolic_keywords:
            f.write(word + "\n")
    with open(os.path.join(args.outdir, "exact_keywords.txt"), "w", encoding="utf-8") as f:
        for word in exact_keywords:
            f.write(word + "\n")
