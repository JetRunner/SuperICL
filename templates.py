def get_input_template(example, dataset_name):
    if dataset_name == "mnli":
        return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    elif dataset_name == "sst2":
        return f"Sentence: {example['sentence']}"
    elif dataset_name == "qnli":
        return f"Question: {example['question']}\nSentence: {example['sentence']}"
    elif dataset_name == "mrpc":
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}"
    elif dataset_name == "qqp":
        return f"Question 1: {example['question1']}\nQuestion 2: {example['question2']}"
    elif dataset_name == "cola":
        return f"Sentence: {example['sentence']}"
    elif dataset_name == "rte":
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}"


def get_plugin_template(example, dataset_name):
    if dataset_name == "mnli":
        return f"{example['premise']} <s> {example['hypothesis']}"
    elif dataset_name == "sst2":
        return f"{example['sentence']}"
    elif dataset_name == "qnli":
        return f"{example['question']} <s> {example['sentence']}"
    elif dataset_name == "mrpc":
        return f"{example['sentence1']} <s> {example['sentence2']}"
    elif dataset_name == "qqp":
        return f"{example['question1']} <s> {example['question2']}"
    elif dataset_name == "cola":
        return f"{example['sentence']}"
    elif dataset_name == "rte":
        return f"{example['sentence1']} <s> {example['sentence2']}"
