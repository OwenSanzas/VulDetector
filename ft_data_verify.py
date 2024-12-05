from collections import defaultdict
import json

data_path = "data/gpt4_fine_tuning_train.jsonl"

with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

print("Num Examples:", len(dataset))
print("First Example:")
for message in dataset[0]:
    print(f"\t{message}: {dataset[0][message]} \n")


format_errors = defaultdict(int)

for ex in dataset:
    if not isinstance(ex, dict):
        format_errors["Examples not formatted as dictionaries"] += 1
        continue

    messages = ex.get("messages", None)

    if not messages:
        format_errors["Missing_msg_list"] += 1
        continue

    for message in messages:
        if 'role' not in message or 'content' not in message:
            format_errors["Missing_'message'_key"] += 1

        if any(k not in ('role', 'content', 'name', 'function_call') for k in message):
            format_errors["unrecognized_key"] += 1

        if message.get('role', None) not in ('system', 'user', 'assistant', 'function'):
            format_errors["unrecognized_role"] += 1

        content = message.get('content', None)
        function_call = message.get('function_call', None)
        if (not content and not function_call) or not isinstance(content, str):
            format_errors["missing_content"] += 1

    if not any(message.get('role', None) == 'assistant' for message in messages):
        format_errors["example_missing_assistant_msg"] += 1


if format_errors:
    print("Format errors:")
    for error, count in format_errors.items():
        print(f"\t{error}: {count}")
else:
    print("No format errors found.")