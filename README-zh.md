# LLM 漂绿检测  

此存储库包含 LLM 漂绿检测项目的代码。  

## 快速开始  

### 安装依赖项  

我们推荐使用 Python 3.10 或更高版本。您可以通过运行以下命令安装依赖项：  

```bash  
pip install -r requirements.txt  
```  

### TF-IDF 和关键词评分的流程  

**1. 自动从 ESG 报告中提取符号关键词和精确关键词**  

运行以下命令，从 ESG 报告中提取符号关键词和精确关键词，其中：  
- ``--data_path`` 是 ESG 报告的路径（包含 `.txt` 文件）。  
- ``--api-key`` 是 DeepSeek API 的 API 密钥。  
- ``--outdir`` 是输出目录的路径。  
- ``--num_reports`` 是要处理的报告数量。  
- ``--use-async`` 指定是否使用异步处理。如果设置了此选项，提取将会*更快*。推荐使用异步处理，同步处理即将被弃用。  

您可以在 ``prompt_templates/keyword_extraction_template.txt`` 中修改关键词提取的提示模板。  

**示例（异步处理）：**  

```bash  
python keyword_extraction.py --data_path ./data --api-key <your_api_key> --outdir ./jieba_wordlist --num_reports 10 --use-async  
```  

**示例（同步处理）：**  

```bash  
python keyword_extraction.py --data_path ./data --api-key <your_api_key> --outdir ./jieba_wordlist --num_reports 10  
```  

**2. 训练评分器**  

通过运行以下命令训练评分器，其中：  
- ``--data_path`` 是 ESG 报告的路径。  
- ``--scoring_method`` 是要使用的评分方法，多个方法以逗号分隔。  
- ``--save_path`` 是保存训练好的评分器的路径。  

**示例：**  

```bash  
python train_scorers.py --data_path ./data --scoring_method kw,tfidf --save_path ./pretrained_scorer  
```  

**3. 对 ESG 报告进行评分**  

通过运行以下命令对 ESG 报告进行评分，其中：  
- ``--data_path`` 是 ESG 报告的路径。  
- ``--scoring_method`` 是要使用的评分方法，多个方法以逗号分隔。  
- ``--outdir`` 是保存评分结果的路径。  
- ``--pretrained_path`` 是训练好的评分器的路径。  

**示例：**  

```bash  
python main_scoring.py --data_path ./data --scoring_method kw,tfidf --outdir ./scoring_results --pretrained_path ./pretrained_scorer  
```  
