import re
import csv
import os
from langchain_community.vectorstores import Chroma
from sklearn.metrics import classification_report
from dotenv import load_dotenv
from database import get_searching_results, load_vector_db
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics import classification_report
from prompts import get_query_prompt, get_vote_prompt
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import OpenAIEmbeddings
from database_non_vul import load_vector_db_non_vul, get_searching_results_non_vul


def load_csv(file_path):
    """读取 CSV 文件返回字典列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def load_existing_results(output_file):
    """加载已有的结果，返回已处理的代码片段集合和结果列表"""
    processed_codes = set()
    existing_results = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_codes.add(row['func_before'])
                existing_results.append(row)
    return processed_codes, existing_results


def append_to_csv(output_file, fieldnames, row_dict):
    """追加写入CSV文件"""
    file_exists = os.path.exists(output_file)
    mode = 'a' if file_exists else 'w'

    with open(output_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def main():
    validation_file = "data/test.csv"
    db_path = 'chroma_db'
    db_path_non_vul = 'chroma_db_non_vul'

    vul_vector_db = load_vector_db(db_path)
    non_vul_vector_db = load_vector_db_non_vul(db_path_non_vul)

    # 加载数据
    val_data = load_csv(validation_file)
    total_samples = len(val_data)

    load_dotenv()

    # 初始化模型
    gpt_4o = ChatOpenAI(
        model="ft:gpt-4o-2024-08-06:personal::AauKt0yR",
        temperature=0.1,
        max_tokens=16384,
        max_retries=2,
    )

    claude = ChatAnthropic(
        model='claude-3-5-sonnet-20240620',
        temperature=0,
        max_tokens=8192,
        max_retries=2,
    )

    gpt_4o_chain = gpt_4o | StrOutputParser()
    claude_chain = claude | StrOutputParser()

    for num in [5]:
        output_file = f"test_results_{num}_contexts3.csv"
        fieldnames = ['func_before', 'true_vul', 'pred_vul', 'true_cwe', 'pred_cwe']

        # 加载已处理的结果
        processed_codes, existing_results = load_existing_results(output_file)
        print(f"Found {len(processed_codes)} previously processed samples")

        true_labels = []
        pred_labels = []
        cwe_predictions = []
        cwe_true = []

        print(f"Testing with {num} retrieved contexts:")

        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )

        for i, row in enumerate(val_data, 1):
            code_snippet = row['func_before']

            # 跳过已处理的样本
            if code_snippet in processed_codes:
                print(f"Skipping already processed sample {i}")
                continue

            print(f"\nProgress: {i}/{total_samples} ({(i / total_samples * 100):.1f}%)")

            true_label = int(row['vul'])
            true_cwe = row['CWE ID']

            try:
                searching_results = get_searching_results(
                    code_snippet,
                    vul_vector_db,
                    embedding_model,
                    res_num=3
                )

                searching_results_non_vul = get_searching_results_non_vul(
                    code_snippet,
                    non_vul_vector_db,
                    embedding_model,
                    res_num=2
                )

            except Exception as e:
                print(f"Error in searching results: {str(e)}")
                continue

            prompt = get_query_prompt(code_snippet, searching_results, searching_results_non_vul)
            response = gpt_4o_chain.invoke(prompt)
            final_ans = claude_chain.invoke(get_vote_prompt(response))

            if 'true' in final_ans.lower():
                pred_label = 1
                cwe_match = re.search(r'CWE-\d+', final_ans)
                cwe_id = cwe_match.group() if cwe_match else None
            else:
                pred_label = 0
                cwe_id = None

            true_labels.append(true_label)
            pred_labels.append(pred_label)
            cwe_predictions.append(cwe_id)
            cwe_true.append(true_cwe)

            # 保存当前结果并实时写入
            result = {
                'func_before': code_snippet,
                'true_vul': true_label,
                'pred_vul': pred_label,
                'true_cwe': true_cwe,
                'pred_cwe': cwe_id
            }
            append_to_csv(output_file, fieldnames, result)
            existing_results.append(result)

            print(f"True Label: {true_label}, Predicted Label: {pred_label}")
            print(f"True CWE: {true_cwe}, Predicted CWE: {cwe_id}")

        # 最终的分类报告
        if true_labels and pred_labels:
            print("\nClassification Report for New Predictions:")
            print(classification_report(true_labels, pred_labels,
                                        target_names=["Non-vulnerable", "Vulnerable"]))

        # 合并所有结果的分类报告
        all_true_labels = [int(r['true_vul']) for r in existing_results]
        all_pred_labels = [int(r['pred_vul']) for r in existing_results]

        print("\nClassification Report for All Results:")
        print(classification_report(all_true_labels, all_pred_labels,
                                    target_names=["Non-vulnerable", "Vulnerable"]))

        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()