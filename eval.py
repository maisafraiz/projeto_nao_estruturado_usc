from qa_metrics.f1 import f1_score_with_precision_recall
import pandas as pd
import json
import numpy as np

df = pd.read_csv("example.csv", names=["número_pregunta", "pregunta", "salida_llm", "nombre_variante"])  

results = []
with open("train.jsonl", "r", encoding="utf-8") as f:
    for i,line in enumerate(f):
        gold = json.loads(line)["answer"]
        if len(gold) >1: ### si la respuesta del ground truth tiene varias partes se concatena en una lista
            gold = ' '.join(gold)
        print(gold)
        pred=  df.iloc[i,:]["salida_llm"]
        print(pred)
        res = f1_score_with_precision_recall(gold, pred)
        results. append(res['f1'])  # dict con “f1”, “precision”, “recall”

print("F1 de la variante:", np.mean(results))  