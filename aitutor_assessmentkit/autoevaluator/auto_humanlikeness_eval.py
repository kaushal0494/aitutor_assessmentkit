import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from aitutor_assessmentkit.autoevaluator import autoeval
from aitutor_assessmentkit.helpers.constants import HUMANLIKENESS_OPENAI_ROBERA_MODEL
from aitutor_assessmentkit.helpers import utils


class AutoHumanlikenessEvaluator(autoeval.AutoEvaluator):
    """
    Evaluates the Humanlikeness of tutor responses using various methods, including model-based 
    and keyword-based approaches, to determine how likely a response is human-written.

    This evaluator supports:
    - **Model-Based Evaluation:** Utilizes the `roberta-large-openai-detector` to compute a Humanlikeness score.
    - **Keyword-Based Methods:** Assesses human-like traits based on predefined linguistic patterns or heuristics.

    These methods aim to measure the quality and naturalness of responses in human-AI interactions.
    """

    def __init__(self, *args, checkpoint: str = HUMANLIKENESS_OPENAI_ROBERA_MODEL, **kwargs) -> None:
        """
        Initialize the AutoHumanlikenessEvaluator object.

        Parameters:
            checkpoint (str): The model checkpoint for computing Humanlikeness scores.

        Notes:
        - **Model-Based Evaluation:**
          - Uses the `roberta-large-openai-detector` model, designed to assess how likely a response is human-like.
          - Model Overview:
            - Released by OpenAI alongside the 1.5B parameter GPT-2 model.
            - Available on Hugging Face: https://huggingface.co/openai-community/roberta-large-openai-detector.
            - Official Report: https://d4mucfpksywv.cloudfront.net/papers/GPT_2_Report.pdf.
          - Characteristics:
            - **Utterance-Level:** Evaluates each response independently of context.
            - **Reference-Free:** Does not require gold-standard references.
          - Scores Interpretation: Higher scores indicate a greater likelihood of being human-written.

        - **Keyword-Based Evaluation:**
          - Focuses on linguistic cues and patterns often associated with human-like responses, such as:
            - **Politeness Indicators:** Evaluates phrases like "thank you" or "please."
            - **Hedging Language:** Checks for terms like "maybe," "I think," or "possibly."
            - **Empathy or Emotional Language:** Detects responses that acknowledge feelings or sentiments.
          - Allows customizable rules to adapt to specific evaluation needs.
        """
        super().__init__(*args, **kwargs)

        # Load tokenizer and model for Humanlikeness evaluation
        self.checkpoint = checkpoint
        self.hum_tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.hum_model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint).to(self.device)

    def _calculate_OGPT2_based_score(self, convs: List[Dict[str, Any]], tutor_model: str) -> Union[List[float], float]:
        """
        Compute Humanlikeness scores using a Roberta-based model.

        Parameters:
            convs (List[Dict[str, Any]]): Conversations data.
            tutor_model (str): The tutor model name.

        Returns:
            List[float]: Scores for each conversation.
            float: Average score.
        """
        scores = []
        for example in convs:
            if utils.should_skip_example(tutor_model, example):
                continue
            response = utils.get_response_text(tutor_model, example)
            inputs = self.hum_tokenizer(response, truncation=True, padding=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.hum_model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=1).cpu()
            score = round(probabilities[:, 1].tolist()[0], 3)
            scores.append(score)
            utils.update_auto_annotation(tutor_model, example, score, 'Humanlikeness_OGPT2')
        return scores, sum(scores) / len(scores)

    def _calculate_heuristic_based_score(self, convs: List[Dict[str, Any]], tutor_model: str) -> Union[List[float], float]:
        """
        Compute Humanlikeness scores using heuristics.

        Parameters:
            convs (List[Dict[str, Any]]): Conversations data.
            tutor_model (str): The tutor model name.

        Returns:
            List[float]: Scores for each conversation.
            float: Average score.
        """
        scores = []
        for example in convs:
            if utils.should_skip_example(tutor_model, example):
                continue
            response = utils.get_response_text(tutor_model, example).lower()
            word_count = len(response.strip().split())
            has_quotes = "tutor" in response

            if any(keyword in response for keyword in ["recheck", "good", "great"]) and word_count <= 3:
                score = 0.5
            elif has_quotes or word_count > 40:
                score = 0.5
            else:
                score = 1.0

            scores.append(score)
            utils.update_auto_annotation(tutor_model, example, score, 'Humanlikeness_Heuristic')
        return scores, sum(scores) / len(scores)
    
    def _get_metric_method(self, metric: str):
        """
        Get the evaluation method based on the metric name.

        Parameters:
            metric (str): Evaluation metric.

        Returns:
            Callable: Evaluation method.
        """
        metric_methods = {
            'Humanlikeness_OGPT2': self._calculate_OGPT2_based_score,
            'Humanlikeness_Heuristic': self._calculate_heuristic_based_score
        }
        if metric not in metric_methods:
            raise ValueError(f"Unsupported metric: {metric}")
        return metric_methods[metric]
    
    def compute(
        self, 
        data: Dict[str, Any] = None, 
        metrics: List[str] = None,
        save: bool = False, 
        file_name: str = 'mrbench_humanlikeness.json'
    ) -> Tuple[Dict[str, Any], List[float]]:
        """
        Compute Humanlikeness scores for tutor models.

        Parameters:
            data (Dict[str, Any]): Input data for evaluation.
            method (List[str]): Evaluation mwtrics to use. Default is ['Humanlikeness_OGPT2'].
            save (bool): Whether to save the results.
            file_name (str): File name for saving results.

        Returns:
            Dict[str, Any]: Final scores.
            List[float]: Collected annotations.
        """
        if data is not None:
            self.data = data

        if metrics is None:
            metrics = ['Humanlikeness_OGPT2']

        print(f"Computing Humanlikeness Scores using {metrics} mtrics(s) for {len(self.data)} examples...")

        final_scores = {metric: {} for metric in metrics}
        collect_annotations = {metric: [] for metric in metrics}

        for metric in metrics:
            metric_method = self._get_metric_method(metric)

            for tutor_name in tqdm(self.tutor_models, desc=f'Calculating {metric} Score for Tutors'):
                anns, score = metric_method(self.data, tutor_name)
                final_scores[metric][tutor_name] = round(score, 3)
                collect_annotations[metric].extend(anns)

            final_scores[metric]['Overall'] = round(
                sum(collect_annotations[metric]) / len(collect_annotations[metric]), 3
            )

        if save:
            utils.save_data(self.output_data_dir, file_name, self.data)

        return final_scores, collect_annotations, self.data

    def list_available_metrics(self) -> pd.DataFrame:
        """
        List available evaluation methods and their descriptions.

        Returns:
            pd.DataFrame: Methods and descriptions.
        """
        methods = {
            "Humanlikeness_OGPT2": "Compute Humanlikeness using a pretrained Roberta model.",
            "Humanlikeness_Heuristic": "Compute Humanlikeness using keyword-based heuristic rules."
        }
        return pd.DataFrame(methods.items(), columns=["Method", "Description"])

    def get_sample_examples_with_scores(
        self, 
        tutor_model: str = 'Expert', 
        num_examples: int = 5, 
        metric: str = 'Humanlikeness_OGPT2'
    ) -> pd.DataFrame:

        """
        Get examples with Humanlikeness scores for a given metric and tutor model.

        Parameters:
            tutor_model (str): The name of the tutor model.
            num_examples (int): The number of examples to display.
            metric (str): The metric to use for evaluation.

        Returns:
            pd.DataFrame: A DataFrame with the examples and their Humanlikeness scores.
        """
        if num_examples > len(self.data):
            raise ValueError(f"Number of examples should be less than or equal to {len(self.data)}")

        metric_method = self._get_metric_method(metric)
        scores, _ = metric_method(self.data[:num_examples], tutor_model)
        return utils.get_sample_dataframe(self.data[:num_examples], tutor_model, scores, dim=metric)
    
    def compare_tutors_scores(
        self, 
        tutor_model1: str = 'Expert', 
        tutor_model2: str = 'GPT4', 
        num_examples: int = 5, 
        metric: str = 'Humanlikeness_OGPT2'
    ) -> pd.DataFrame:

        """
        Compare the Humanlikeness scores of two tutor models for a specific metric.

        Parameters:
            tutor_model1 (str): The name of the first tutor model.
            tutor_model2 (str): The name of the second tutor model.
            num_examples (int): The number of examples to display.
            metric (str): The metric to use for comparison.

        Returns:
            pd.DataFrame: A DataFrame with the examples and their Humanlikeness scores.
        """
        if num_examples > len(self.data):
            raise ValueError(f"Number of examples should be less than or equal to {len(self.data)}")

        metric_method = self._get_metric_method(metric)
        scores1, _ = metric_method(self.data[:num_examples], tutor_model1)
        scores2, _ = metric_method(self.data[:num_examples], tutor_model2)

        return utils.get_sample_dataframe(self.data[:num_examples], tutor_model1, scores1,
                                          tutor_model2=tutor_model2, scores2=scores2, dim=metric)