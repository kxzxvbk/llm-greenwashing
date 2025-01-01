import os
import argparse
import random
import time
import warnings
from typing import Set

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


def process_chunk(chunk: str, api_agent: DeepseekAPIClient):
    try:
        prompt = fill_in_template(keyword_extraction_template, document=chunk)
        query = [{"role": "user", "content": prompt}]
        response = api_agent.generate(query)
        response = eval(extract_json(response))
        return (set(response['象征性环境行动关键词']), set(response['实际性环境行动关键词']))
    except Exception as e:
        print(f'Error processing chunk: {e}')
        return (set(), set())


def process_file(file_path: str, api_agent: DeepseekAPIClient, symbolic_keywords: Set[str], exact_keywords: Set[str]):
    with open(file_path, encoding='utf-8') as f:
        file_content = f.read()
    chunked_content = split_content(file_content)
    
    # Add progress bar for chunks within each file
    results = [process_chunk(chunk, api_agent) for chunk in chunked_content]
    
    for symbolic, exact in results:
        symbolic_keywords.update(symbolic)
        exact_keywords.update(exact)


async def async_process_chunk(chunk: str, api_agent: DeepseekAPIClient):
    try:
        prompt = fill_in_template(keyword_extraction_template, document=chunk)
        query = [{"role": "user", "content": prompt}]
        response = await api_agent.async_generate(query)
        response = eval(extract_json(response))
        return (set(response['象征性环境行动关键词']), set(response['实际性环境行动关键词']))
    except Exception as e:
        print(f'Error processing chunk: {e}')
        return (set(), set())


async def async_process_file(file_path: str, api_agent: DeepseekAPIClient, symbolic_keywords: Set[str], exact_keywords: Set[str]):
    with open(file_path, encoding='utf-8') as f:
        file_content = f.read()
    chunked_content = split_content(file_content)
    
    # Add progress bar for chunks within each file
    tasks = [async_process_chunk(chunk, api_agent) for chunk in chunked_content]
    results = await asyncio.gather(*tasks)
    
    for symbolic, exact in results:
        symbolic_keywords.update(symbolic)
        exact_keywords.update(exact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data', required=False, help='The path to the corpus.')
    parser.add_argument('--api-key', required=True, help='The API key for the DeepSeek API.')
    parser.add_argument('--outdir', default='./jieba_wordlist', required=False, help='The path to save the extracted keywords.')
    parser.add_argument('--num_reports', default=10, type=int, required=False, help='The number of reports to extract keywords from.')
    parser.add_argument('--use-async', action='store_true', help='Use asynchronous processing.')
    args = parser.parse_args()

    if args.use_async:
        try:
            import asyncio
        except ImportError:
            raise ImportError("Async processing requires the asyncio module. Please install it using 'pip install asyncio'.")
        print("Using asynchronous processing.")
    else:
        warnings.warn("Synchronous processing is deprecated. Please use --use-async instead.")
        print("Using synchronous processing.")
    
    start_time = time.time()
    root_path = args.data_path

    api_agent = DeepseekAPIClient(api_key=args.api_key)
    symbolic_keywords, exact_keywords = preload_keywords()

    target_files = os.listdir(root_path)
    target_files = random.sample(target_files, args.num_reports)

    async def async_main():
        # Add progress bar for overall file processing
        tasks = [async_process_file(os.path.join(root_path, file), api_agent, symbolic_keywords, exact_keywords) for file in target_files]
        await asyncio.gather(*tasks)
    
    def main():
        for file in target_files:
            process_file(os.path.join(root_path, file), api_agent, symbolic_keywords, exact_keywords)

    if args.use_async:
        asyncio.run(async_main())
    else:
        main()

    if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        
    # Save the extracted keywords.
    with open(os.path.join(args.outdir, "symbolic_keywords.txt"), "w", encoding="utf-8") as f:
        for word in symbolic_keywords:
            f.write(word + "\n")
    with open(os.path.join(args.outdir, "exact_keywords.txt"), "w", encoding="utf-8") as f:
        for word in exact_keywords:
            f.write(word + "\n")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
