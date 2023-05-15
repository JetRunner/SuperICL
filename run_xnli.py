import torch
import datasets
import transformers
import random
import time
import numpy as np
from tqdm.auto import tqdm, trange
import argparse

from utils import gpt3_complete_with_auto_reduce


def convert_label(label):
    return {"LABEL_2": "contradiction", "LABEL_1": "neutral", "LABEL_0": "entailment"}[
        label
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--model_name", type=str, default="XLM-V", help="Name of model for prompts"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en,ar,bg,de,el,es,fr,hi,ru,sw,th,tr,ur,vi,zh",
        help="Languages to test, comma separated",
    )
    parser.add_argument(
        "--num_examples", type=int, default=16, help="Number of in-context examples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--run_icl", action="store_true", default=True, help="Run ICL baseline"
    )
    parser.add_argument(
        "--run_plugin_model",
        action="store_true",
        default=True,
        help="Run plugin model baseline",
    )
    parser.add_argument(
        "--run_supericl", action="store_true", default=True, help="Run SuperICL"
    )
    parser.add_argument(
        "--sleep_time", type=float, default=0.5, help="Sleep time between GPT API calls"
    )
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)


plugin_model = transformers.pipeline("text-classification", model=args.model_path)
langs = args.lang.split(",")

dataset = datasets.load_dataset("xnli", langs[0])
label_map = dataset["train"].features["label"].names

for lang_idx, lang in enumerate(langs):
    if lang_idx != 0:
        dataset = datasets.load_dataset("xnli", lang)
    train = dataset["train"].shuffle().select(range(args.num_examples))

    if args.run_icl:
        in_context_prompt = ""
        for example in train:
            in_context_prompt += f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nLabel: {label_map[example['label']]}\n\n"

        total_icl = 0
        correct_icl = 0
        for example in tqdm(dataset["test"]):
            valid_prompt = (
                in_context_prompt
                + f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nLabel: "
            )
            response = gpt3_complete_with_auto_reduce(
                engine="text-davinci-003",
                prompt=valid_prompt,
                temperature=1,
                max_tokens=10,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0,
                best_of=1,
                stop=None,
            )
            time.sleep(args.sleep_time)
            result = response["choices"][0]["text"].strip()
            total_icl += 1
            if result == label_map[example["label"]]:
                correct_icl += 1

        print(f"Language: {lang}, ICL Accuracy: {correct_icl / total_icl}")

    if args.run_plugin_model:
        total_plugin_model = 0
        correct_plugin_model = 0
        for example in tqdm(dataset["test"]):
            text = f"{example['premise']} <s> {example['hypothesis']}"
            result = convert_label(plugin_model(text)[0]["label"])
            total_plugin_model += 1
            if result == label_map[example["label"]]:
                correct_plugin_model += 1

        print(
            f"Language: {lang}, Plugin Model Accuracy: {correct_plugin_model / total_plugin_model}"
        )

    if args.run_supericl:
        in_context_distill_prompt = ""
        for example in train:
            text = f"{example['premise']} <s> {example['hypothesis']}"
            plugin_model_res = plugin_model(text)[0]
            plugin_model_label = convert_label(plugin_model_res["label"])
            plugin_model_confidence = round(plugin_model_res["score"], 2)
            in_context_distill_prompt += f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\n{args.model_name} Prediction: {plugin_model_label} (Confidence: {plugin_model_confidence})\nLabel: {label_map[example['label']]}\n\n"

        total_distill = 0
        correct_distill = 0
        for example in tqdm(dataset["test"]):
            text = f"{example['premise']} <s> {example['hypothesis']}"
            plugin_model_res = plugin_model(text)[0]
            plugin_model_label = convert_label(plugin_model_res["label"])
            plugin_model_confidence = round(plugin_model_res["score"], 2)
            valid_prompt = (
                in_context_distill_prompt
                + f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\n{args.model_name} Prediction: {plugin_model_label} (Confidence: {plugin_model_confidence})\nLabel: "
            )
            response = gpt3_complete_with_auto_reduce(
                engine="text-davinci-003",
                prompt=valid_prompt,
                temperature=1,
                max_tokens=10,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0,
                best_of=1,
                stop=None,
            )
            time.sleep(args.sleep_time)
            result = response["choices"][0]["text"].strip()
            total_distill += 1
            if result == label_map[example["label"]]:
                correct_distill += 1

        print(f"Language: {lang}, SuperICL Accuracy: {correct_distill / total_distill}")
