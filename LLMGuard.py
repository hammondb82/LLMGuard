# code was adapted from https://llm-guard.com/tutorials/notebooks/langchain/#what-is-lcel to use
# Mistal-7B-instruct-v0.2 as the LLM instead of an OpenAI model.

import os
import json
import requests
import pandas as pd
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
from langchain_community.llms import HuggingFaceEndpoint
from LLMGuardPromptChain import LLMGuardPromptChain, LLMGuardPromptException
from LLMGuardOutputChain import LLMGuardOutputChain, LLMGuardOutputException
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
from llm_guard.vault import Vault
import logging
from llm_guard.input_scanners.toxicity import MatchType
import time
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.schema.output_parser import StrOutputParser
use_onnx = False

logger = logging.getLogger(__name__)

vault = Vault()

llm_guard_prompt_scanner = LLMGuardPromptChain(
    vault=vault,
    scanners={
        "Anonymize": {"use_faker": True, "use_onnx": use_onnx},
        "BanTopics": {"topics": ["violence", "Sexuality", "Racism", "Suicide", "Drugs"], "threshold": 0.7, "use_onnx": use_onnx},
        "Language": {"valid_languages": ["en"], "use_onnx": use_onnx},
        "PromptInjection": {"threshold": 0.95, "use_onnx": use_onnx},
        "Regex": {"patterns": ["Bearer [A-Za-z0-9-._~+/]+"]},
        "Secrets": {"redact_mode": "all"},
        "Sentiment": {"threshold": -0.05},
        "TokenLimit": {"limit": 4096},
        "Toxicity": {"threshold": 0.8, 'match_type': MatchType.SENTENCE, "use_onnx": use_onnx},
    },
    # scanners_ignore_errors=[
    #     "Anonymize",
    #     "BanSubstrings",
    #     "Regex",
    #     "Secrets",
    #     "TokenLimit",
    #     "PromptInjection",
    # ],  # These scanners redact, so I can skip them from failing the prompt
)


llm = HuggingFaceEndpoint(
        repo_id=repo_id, token=HUGGINGFACEHUB_API_TOKEN
    )

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You're a helpful assistant"
        ),
        HumanMessagePromptTemplate.from_template("{sanitized_input}"),
    ]
)

#Change this path based on what dataset you want to run
file_path = "base_set/prompts/test_script.json"
input_prompts = []
qid = []

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)


    for entry in data:
        input_prompts.append(entry.get('question'))
        qid.append(entry.get('qid'))

# allows use of csv files
# data_csv = pd.read_csv(file_path)
# input_prompts = data_csv['prompt'].tolist()
# qid = data_csv['id_v2'].tolist()


llm_guard_output_scanner = LLMGuardOutputChain(
    vault=vault,
    scanners={
        "BanTopics": {"topics": ["violence", "Sexuality", "Racism", "Suicide", "Drugs"], "threshold": 0.8, "use_onnx": use_onnx},
        "Bias": {"threshold": 0.75, "use_onnx": use_onnx},
        "Deanonymize": {},
        "FactualConsistency": {"minimum_score": 0.5, "use_onnx": use_onnx},
        # "JSON": {"required_elements": 0, "repair": True},
        "Language": {
            "valid_languages": ["en"],
            "threshold": 0.5,
            "use_onnx": use_onnx,
        },
        "LanguageSame": {"use_onnx": use_onnx},
        "MaliciousURLs": {"threshold": 0.75, "use_onnx": use_onnx},
        "NoRefusal": {"threshold": 0.5, "use_onnx": use_onnx},
        "Regex": {
            "patterns": ["Bearer [A-Za-z0-9-._~+/]+"],
        },
        # "Relevance": {"threshold": 0.5, "use_onnx": use_onnx},
        "Sensitive": {"redact": False, "use_onnx": use_onnx},
        "Sentiment": {"threshold": -0.05},
        "Toxicity": {"threshold": 0.7, "use_onnx": use_onnx},
    },
    # scanners_ignore_errors=["BanSubstrings", "Regex", "Sensitive"],
)

start_time = time.time()
results = []

# I made most of this code below this line to catch any errors and exceptions raised by LLM-Guard.
def attempt_request(input_prompt, retries=3):
    for attempt in range(retries):
        try:
            guarded_chain = (
                llm_guard_prompt_scanner
                | prompt
                | llm
                | (lambda ai_message: llm_guard_output_scanner.scan(input_prompt, ai_message))
                | StrOutputParser()
            )

            result = guarded_chain.invoke({"input": input_prompt})
            return result
        except requests.exceptions.ConnectionError as e:
            if attempt < retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)
            else:
                raise e
        except requests.exceptions.HTTPError as e:
            if e.response:
                if e.response.status_code == 502:
                    print(f"Attempt {attempt + 1} failed with 502 Bad Gateway, retrying...")
                    time.sleep(2 ** attempt)
                elif e.response.status_code == 503:
                    print(f"Attempt {attempt + 1} failed with 503 Service Unavailable, retrying...")
                    time.sleep(2 ** (attempt + 1))
                elif e.response.status_code == 504:
                    print(f"Attempt {attempt + 1} failed with 504 Gateway Timeout, retrying...")
                    time.sleep(2 ** attempt)
                else:

                    print(f"HTTP error with status code {e.response.status_code} encountered.")
                    raise
                continue
            else:
                print("No response received from server.")
                raise

for i, input_prompt in enumerate(input_prompts):
    try:
        result = attempt_request(input_prompt)
        results.append({"qid": qid[i], "answer": result})
        print(i)

    except LLMGuardPromptException as e:
        print('prompt')
        results.append({
            "qid": qid[i],
            "answer": f"Input: {str(e)}",
            "input": input_prompt
        })

    except LLMGuardOutputException as e:
        results.append({
            "qid": qid[i],
            "answer": f"Output: {str(e)}",
            "input": input_prompt
        })

    except Exception as e:
        results.append({
            "qid": qid[i],
            "answer": "Connection error occurred.",
            "input": input_prompt
        })


save_file_path = "results.json"
with open(save_file_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)
end_time = time.time()
print(end_time - start_time)