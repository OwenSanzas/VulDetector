import pandas as pd
import re
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.metrics import classification_report
from dotenv import load_dotenv
from database import get_searching_results, load_vector_db
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from prompts import get_query_prompt, get_vote_prompt
from langchain_anthropic import ChatAnthropic


validation_file = "data/val.csv"
db_path = 'chroma_db'
val_df = pd.read_csv(validation_file)

load_dotenv()


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

num_res = [3, 5, 10]

vote = [1, 3, 5]

v = 1

for num in num_res:
    true_labels = []
    pred_labels = []
    cwe_predictions = []
    cwe_true = []

    print(f"Testing with {num} retrieved contexts:")
    cnt = 1
    for _, row in val_df.iterrows():
        print(f"Testing {cnt}...")
        cnt += 1

        print(row)

        code_snippet = row['func_before']
        true_label = row['vul']
        true_cwe = row['CWE ID']

        try:
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-large"
            )

            searching_results = get_searching_results(code_snippet, load_vector_db('chroma_db'), embedding_model, res_num=3)
        except Exception as e:
            print(f"Error in searching results: {str(e)}")

        print(f"Searching results: {searching_results}")
        prompt = get_query_prompt(code_snippet, searching_results)

        current_ans = []

        response = gpt_4o_chain.invoke(prompt)
        current_ans.append(response)

        all_results = "\n".join(current_ans)
        final_ans = claude_chain.invoke(get_vote_prompt(all_results))

        if 'true' in final_ans:
            pred_label = 1
            # use regex to extract the CWE ID from the response output CWE-xxx
            cwe_id = re.search(r'CWE-\d+', final_ans).group()
        else:
            pred_label = 0
            cwe_id = None

        true_labels.append(true_label)
        pred_labels.append(pred_label)
        cwe_predictions.append(cwe_id)
        cwe_true.append(true_cwe)

        print(f"True Label: {true_label}, Predicted Label: {pred_label}")
        print(f"True CWE: {true_cwe}, Predicted CWE: {cwe_id}")

    # 计算分类性能指标
    print("\nClassification Report for Vul Prediction:")
    print(classification_report(true_labels, pred_labels, target_names=["Non-vulnerable", "Vulnerable"]))

    # 保存结果到 CSV 文件
    output_file = f"validation_results_{num}_contexts.csv"
    results_df = pd.DataFrame({
        "func_before": val_df["func_before"],
        "true_vul": true_labels,
        "pred_vul": pred_labels,
        "true_cwe": val_df["CWE ID"],
        "pred_cwe": cwe_predictions,
    })

    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}.")

















