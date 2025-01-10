import evaluate
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from wordllama import WordLlamaInference
from numpy import ndarray

bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
model_name = "sentence-transformers/sentence-t5-base"
model = SentenceTransformer(model_name)

def unpack_y_true(y_true):
    return [i[0] for i in y_true]

def st_sim(y_pred: List[str], y_true: List[List[str]]) -> ndarray:
        embd1 = model.encode(y_pred, show_progress_bar=False)
        embd2 = model.encode(unpack_y_true(y_true), show_progress_bar=False)
        return model.similarity_pairwise(embd1, embd2).numpy()
    

def compute_metrics(y_pred: List[str], y_true: List[List[str]]) -> Dict:

    bleu_score = bleu.compute(predictions=y_pred, references=y_true)

    bertscore_score = bertscore.compute(
        predictions=y_pred,
        references=y_true,
        lang="en",
        model_type="distilbert-base-uncased",
        device="cuda",
    )

    st_similarities = st_sim(y_pred, y_true)

    return {
        "bleu": bleu_score,
        "bertscore": bertscore_score,
        "st_similarities": st_similarities
    }