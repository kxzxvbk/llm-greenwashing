import openai
import re
import copy
import warnings
import time


class DeepseekAPIClient:

    def __init__(self, api_key=None):
        self.default_generate_cfg = dict(temperature=0.9, top_p=0.7, frequency_penalty=0, presence_penalty=0, stop=None)
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def generate(self, history, max_retry=5, **kwargs):
        # Generate the response from the LLM.
        generate_cfg = copy.deepcopy(self.default_generate_cfg)
        generate_cfg.update(kwargs)

        for _ in range(max_retry):
            try:
                response_i = self.client.chat.completions.create(
                    model='deepseek-chat', messages=history, **generate_cfg
                )
            except Exception as e:
                warnings.warn(e)
                time.sleep(3)
                continue
            reply_i = response_i.choices[0].message.content.strip()
            return reply_i
        assert False, "Maximum retry exceeds ..."


def fill_in_template(template, **kwargs):
    # Fill in the template with the given arguments.
    for key, value in kwargs.items():
        if not isinstance(key, str):
            key = str(key)
        if not isinstance(value, str):
            value = str(value)
        assert '{{' + key + '}}' in template, f'Argument {key} is not in the template.'
        template = template.replace('{{' + key + '}}', value)
    return template


def extract_json(text):
    # Extract the JSON string from the text.
    match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if match:
        json_str = match.group(1)
        return json_str
    else:
        return None
