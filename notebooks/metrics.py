import evaluate
from typing import List, Dict

bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

def compute_metrics(y_pred: List[str], y_true: List[List[str]]) -> Dict:

    bleu_score = bleu.compute(predictions=y_pred, references=y_true)

    bertscore_score = bertscore.compute(
        predictions=y_pred,
        references=y_true,
        lang="en",
        model_type="distilbert-base-uncased",
        device="cuda",
    )

    return {
        "bleu": bleu_score,
        "bertscore": bertscore_score,
    }