import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from aitutor_assessmentkit.autoevaluator import autoeval
from aitutor_assessmentkit.helpers.constants import TUTORTONE_MODEL
from aitutor_assessmentkit.helpers import utils


class AutoTutorToneEvaluator(autoeval.AutoEvaluator):
    """
    A class to evaluate Tutor Tone in tutor responses.
    """

    def __init__(self, checkpoint: str = TUTORTONE_MODEL, *args, **kwargs) -> None:
        """
        Initialize the AutoTutorToneEvaluator object, inheriting from AutoEvaluator.

        Parameters:
            checkpoint (str): The checkpoint for the RoBERTa model fine-tuned on the empathy dataset.

        Notes:
        - The Tutor Tone score is computed using a binary classifier fine-tuned on an empathy classification dataset.
        - A higher score indicates better performance in conveying empathetic tone.
        """
        super().__init__(*args, **kwargs)

        self.checkpoint = checkpoint
        self.tutortone_tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.tutortone_model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint).to(self.device)

    def _calculate_tutortone_score(
        self, convs: List[Dict[str, Any]], tutor_model: str
    ) -> Union[List[float], float]:
        """
        Calculate the Tutor Tone score for conversations.

        Parameters:
            convs (List[Dict[str, Any]]): List of conversations.
            tutor_model (str): The name of the tutor model.

        Returns:
            List[float]: Individual scores for each conversation.
            float: Average score across all conversations.
        """
        all_tutortone_scores = []

        for example in convs:
            if utils.should_skip_example(tutor_model, example):
                continue

            response = utils.get_response_text(tutor_model, example)
            inputs = self.tutortone_tokenizer(
                str(response), padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.tutortone_model(**inputs)

            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1).cpu()
            tutortone_score = round(probabilities[:, 1].tolist()[0], 3)
            all_tutortone_scores.append(tutortone_score)

            utils.update_auto_annotation(tutor_model, example, tutortone_score, 'Tutor_Tone_FTRoBERTa')

        return all_tutortone_scores, sum(all_tutortone_scores) / len(all_tutortone_scores)

    def compute(
        self, 
        data: Dict[str, Any] = None, 
        metrics: List[str] = None, 
        save: bool = False, 
        file_name: str = 'mrbench_tutortone.json'
    ) -> Tuple[Dict[str, float], List[float]]:
        """
        Evaluate Tutor Tone scores for each model in the dataset.

        Parameters:
            data (Dict[str, Any]): Input data for evaluation.
            metrics (List[str]): List of metrics to evaluate. Default is ['Tutor_Tone_FTRoBERTa'].
            save (bool): Whether to save the computed results to a file.
            file_name (str): Name of the output file.

        Returns:
            Tuple[Dict[str, float], List[float]]: Final scores and annotations.
        """
        if data is not None:
            self.data = data

        if metrics is None:
            metrics = ['Tutor_Tone_FTRoBERTa']  # Default metric

        print(f"Computing Tutor Tone Scores using {metrics} metric(s) for {len(self.data)} examples...")

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

    def _get_metric_method(self, metric: str):
        """
        Retrieve the method for the given metric.

        Parameters:
            metric (str): The metric name.

        Returns:
            Callable: The method corresponding to the metric.
        """
        metric_methods = {
            'Tutor_Tone_FTRoBERTa': self._calculate_tutortone_score
        }
        if metric not in metric_methods:
            raise ValueError(f"Unsupported metric: {metric}")
        return metric_methods[metric]
    
    def list_available_metrics(self) -> pd.DataFrame:
        """
        List available evaluation methods and their descriptions.

        Returns:
            pd.DataFrame: Methods and descriptions.
        """
        methods = {
            "Tutor_Tone_FTRoBERTa":"Tutor Tone score using a fine-tuned RoBERTa model on the empathy dataset." 
        }
        return pd.DataFrame(methods.items(), columns=["Method", "Description"])

    def get_sample_examples_with_scores(
        self, 
        tutor_model: str = 'Expert', 
        num_examples: int = 5, 
        metric: str = 'Tutor_Tone_FTRoBERTa'
    ) -> pd.DataFrame:
        """
        Get examples with Tutor Tone scores for a given metric and tutor model.

        Parameters:
            tutor_model (str): The name of the tutor model.
            num_examples (int): The number of examples to display.
            metric (str): The metric to use for evaluation.

        Returns:
            pd.DataFrame: A DataFrame with the examples and their Tutor Tone scores.
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
        metric: str = 'Tutor_Tone_FTRoBERTa'
    ) -> pd.DataFrame:
        """
        Compare the Tutor Tone scores of two tutor models for a specific metric.

        Parameters:
            tutor_model1 (str): The name of the first tutor model.
            tutor_model2 (str): The name of the second tutor model.
            num_examples (int): The number of examples to display.
            metric (str): The metric to use for comparison.

        Returns:
            pd.DataFrame: A DataFrame with the examples and their Tutor Tone scores.
        """
        if num_examples > len(self.data):
            raise ValueError(f"Number of examples should be less than or equal to {len(self.data)}")

        metric_method = self._get_metric_method(metric)
        scores1, _ = metric_method(self.data[:num_examples], tutor_model1)
        scores2, _ = metric_method(self.data[:num_examples], tutor_model2)

        return utils.get_sample_dataframe(self.data[:num_examples], tutor_model1, scores1,
                                          tutor_model2=tutor_model2, scores2=scores2, dim=metric)