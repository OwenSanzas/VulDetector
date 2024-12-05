import pandas as pd
import numpy as np
from collections import defaultdict


def analyze_cwe_predictions(df):
    """分析CWE预测准确性"""
    # 只分析vul=1的样本
    vul_samples = df[df['true_vul'] == 1]

    # 计算CWE预测准确率
    correct_cwe = sum(vul_samples['true_cwe'] == vul_samples['pred_cwe'])
    total_vul = len(vul_samples)

    print(f"CWE Prediction Accuracy for Vulnerable Samples:")
    print(f"Correct predictions: {correct_cwe}")
    print(f"Total vulnerable samples: {total_vul}")
    print(f"Accuracy: {correct_cwe / total_vul:.3f}")

    # 按CWE类型统计准确率
    cwe_performance = defaultdict(lambda: {'correct': 0, 'total': 0})

    for _, row in vul_samples.iterrows():
        true_cwe = row['true_cwe']
        pred_cwe = row['pred_cwe']

        cwe_performance[true_cwe]['total'] += 1
        if true_cwe == pred_cwe:
            cwe_performance[true_cwe]['correct'] += 1

    # 计算每种CWE的准确率并排序
    cwe_accuracy = {}
    for cwe, stats in cwe_performance.items():
        accuracy = stats['correct'] / stats['total']
        cwe_accuracy[cwe] = {
            'accuracy': accuracy,
            'correct': stats['correct'],
            'total': stats['total']
        }

    # 按准确率排序

    sorted_cwe = sorted(cwe_accuracy.items(),
                        key=lambda x: (x[1]['accuracy'], x[1]['total']),
                        reverse=True)

    print("\nCWE Performance Ranking:")
    print("CWE-ID | Accuracy | Correct | Total")
    print("-" * 40)
    for cwe, stats in sorted_cwe:
        print(f"{cwe:6} | {stats['accuracy']:.3f}   | {stats['correct']:7d} | {stats['total']:5d}")

    return cwe_accuracy


# 分析测试集结果
print("Test Set Analysis:")
tf = pd.read_csv('test_results_5_contexts3_3.csv')
cwe_results = analyze_cwe_predictions(tf)