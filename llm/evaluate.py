import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np

class LLMEvaluator:
    """Evaluation metrics for LLMs."""

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate_perplexity(self, texts):
        """Calculate perplexity on texts."""
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item() * inputs['input_ids'].shape[1]
                total_tokens += inputs['input_ids'].shape[1]

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        return perplexity

    def evaluate_bleu(self, references, predictions):
        """Calculate BLEU score."""
        scores = []
        for ref, pred in zip(references, predictions):
            ref_tokens = ref.split()
            pred_tokens = pred.split()
            score = sentence_bleu([ref_tokens], pred_tokens)
            scores.append(score)
        return np.mean(scores)

    def evaluate_rouge(self, references, predictions):
        """Calculate ROUGE scores."""
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for ref, pred in zip(references, predictions):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }

    def evaluate_all(self, test_data):
        """Run all evaluations."""
        references = []
        predictions = []

        for item in test_data:
            prompt = item['prompt']
            reference = item['reference']

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=100)
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            references.append(reference)
            predictions.append(prediction)

        results = {
            'bleu': self.evaluate_bleu(references, predictions),
            'rouge': self.evaluate_rouge(references, predictions)
        }

        return results
