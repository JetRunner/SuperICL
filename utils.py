import importlib
import openai
from retry import retry
import time
import api_config

openai.api_type = api_config.API_TYPE
openai.api_base = api_config.API_BASE
openai.api_version = api_config.API_VERSION
openai.api_key = api_config.API_KEY


def remove_last_icl_example(prompt):
    split_prompt = prompt.split("\n\n")
    split_prompt.pop(-2)
    return "\n\n".join(split_prompt)


@retry(delay=60, backoff=2, tries=5)
def gpt3_complete_with_auto_reduce(sleep_time, *args, **kwargs):
    """
    This function is used to automatically reduce the number of examples if it is too long for GPT API.
    """
    is_too_long = True

    while is_too_long:
        try:
            res = openai.Completion.create(*args, **kwargs)
            is_too_long = False
        except:
            kwargs["prompt"] = remove_last_icl_example(kwargs["prompt"])
            time.sleep(sleep_time)
            continue
        return res


@retry(delay=60, backoff=2, tries=5)
def gpt3_complete(*args, **kwargs):
    res = openai.Completion.create(*args, **kwargs)
    return res
