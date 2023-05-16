import torch
import datasets
import transformers
import random
import time
import numpy as np
from tqdm.auto import tqdm, trange
import argparse
from sklearn.metrics import matthews_corrcoef, accuracy_score

from utils import gpt3_complete
from templates import get_input_template, get_plugin_template


def convert_label(label, label_list):
    if label.startswith("LABEL_"):
        return label_list[int(label.split("_")[-1])]
    else:
        return label.lower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="RoBERTa-Large",
        help="Name of model for prompts",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnli-m", "mnli-mm", "sst2", "qnli", "mrpc", "qqp", "cola", "rte"],
        help="Dataset to test on",
    )
    parser.add_argument(
        "--num_examples", type=int, default=32, help="Number of in-context examples"
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
    parser.add_argument(
        "--explanation", action="store_true", default=False, help="Run with explanation"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    plugin_model = transformers.pipeline("text-classification", model=args.model_path)
    print(f"Loaded model {args.model_path} with name {args.model_name}")
    print(f"Testing on dataset: {args.dataset}")

    dataset_name = args.dataset.split("-")[0]
    dataset = datasets.load_dataset("glue", dataset_name)
    label_list = dataset["train"].features["label"].names

    train = dataset["train"].shuffle().select(range(args.num_examples))
    test = (
        dataset["validation"]
        if not args.dataset.startswith("mnli")
        else dataset[
            "validation" + {"m": "_matched", "mm": "_mismatched"}[args.dataset[-1]]
        ]
    )

    if args.run_icl:
        in_context_prompt = ""
        for example in train:
            in_context_prompt += f"{get_input_template(example, dataset_name)}\nLabel: {label_list[example['label']]}\n\n"

        icl_predictions = []
        icl_ground_truth = []
        for example in tqdm(test):
            valid_prompt = (
                in_context_prompt
                + f"{get_input_template(example, dataset_name)}\nLabel: "
            )
            response = gpt3_complete(
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
            icl_predictions.append(response["choices"][0]["text"].strip())
            icl_ground_truth.append(label_list[example["label"]])

        if dataset_name == "cola":
            print(
                f"ICL Matthews Corr: {matthews_corrcoef(icl_predictions, icl_ground_truth)}"
            )
        else:
            print(f"ICL Accuracy: {accuracy_score(icl_predictions, icl_ground_truth)}")

    if args.run_plugin_model:
        plugin_model_predictions = []
        plugin_model_ground_truth = []
        for example in tqdm(test):
            plugin_model_label = convert_label(
                plugin_model(get_plugin_template(example, dataset_name))[0]["label"],
                label_list,
            )
            plugin_model_predictions.append(plugin_model_label)
            plugin_model_ground_truth.append(label_list[example["label"]])

        if dataset_name == "cola":
            print(
                f"Plugin Model Matthews Corr: {matthews_corrcoef(plugin_model_predictions, plugin_model_ground_truth)}"
            )
        else:
            print(
                f"Plugin Model Accuracy: {accuracy_score(plugin_model_predictions, plugin_model_ground_truth)}"
            )

    if args.run_supericl:
        in_context_supericl_prompt = ""
        for example in train:
            plugin_input = get_plugin_template(example, dataset_name)
            plugin_model_result = plugin_model(plugin_input)[0]
            plugin_model_label = convert_label(plugin_model_result["label"], label_list)
            plugin_model_confidence = round(plugin_model_result["score"], 2)
            in_context_supericl_prompt += f"{get_input_template(example, dataset_name)}\n{args.model_name} Prediction: {plugin_model_label} (Confidence: {plugin_model_confidence})\nLabel: {label_list[example['label']]}\n\n"

        supericl_predictions = []
        supericl_ground_truth = []
        for example in tqdm(test):
            plugin_input = get_plugin_template(example, dataset_name)
            plugin_model_result = plugin_model(plugin_input)[0]
            plugin_model_label = convert_label(plugin_model_result["label"], label_list)
            plugin_model_confidence = round(plugin_model_result["score"], 2)
            valid_prompt = f"{get_input_template(example, dataset_name)}\n{args.model_name} Prediction: {plugin_model_label} (Confidence: {plugin_model_confidence})\nLabel: "
            response = gpt3_complete(
                engine="text-davinci-003",
                prompt=in_context_supericl_prompt + valid_prompt,
                temperature=1,
                max_tokens=10,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0,
                best_of=1,
                stop=None,
            )
            time.sleep(args.sleep_time)
            supericl_prediction = response["choices"][0]["text"].strip()
            supericl_ground_label = label_list[example["label"]]

            supericl_predictions.append(supericl_prediction)
            supericl_ground_truth.append(supericl_ground_label)

            if args.explanation and supericl_prediction != plugin_model_label:
                explain_prompt = (
                    in_context_supericl_prompt
                    + valid_prompt
                    + "\nExplanation for overriding the prediction:"
                )
                response = gpt3_complete(
                    engine="text-davinci-003",
                    prompt=explain_prompt,
                    temperature=1,
                    max_tokens=100,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    best_of=1,
                    stop=None,
                )
                print(f"\n{valid_prompt + supericl_prediction}")
                print(f"Explanation: {response['choices'][0]['text'].strip()}\n")

        if dataset_name == "cola":
            print(
                f"SuperICL Matthews Corr: {matthews_corrcoef(supericl_predictions, supericl_ground_truth)}"
            )
        else:
            print(
                f"SuperICL Accuracy: {accuracy_score(supericl_predictions, supericl_ground_truth)}"
            )
