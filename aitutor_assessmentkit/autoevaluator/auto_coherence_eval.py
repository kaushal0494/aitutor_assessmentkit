import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from evaluate import load
bertscore = load("bertscore")

from aitutor_assessmentkit.autoevaluator import autoeval
from aitutor_assessmentkit.helpers import utils
from aitutor_assessmentkit.helpers.constants import COHERENCE_NLI_MODEL


class AutoCoherenceEvaluator(autoeval.AutoEvaluator):
    """
    A class to evaluate the coherence of tutor responses using multiple metrics.
    Provides flexibility in evaluating coherence based on different approaches, 
    such as BERTScore and Natural Language Inference (NLI).
    """

    def __init__(self, checkpoint: str=COHERENCE_NLI_MODEL, *args, **kwargs) -> None:
        """
        Initialize the AutoCoherenceEvaluator object, inheriting from AutoEvaluator.

        Parameters:
            args: Positional arguments passed to the parent AutoEvaluator class.
            kwargs: Keyword arguments passed to the parent AutoEvaluator class.

        Notes:
        - This class supports multiple coherence metrics, including:
            1. Coherence_BERT: Uses BERTScore to evaluate the similarity between 
               the student's last utterances and the tutor's response.
            2. Coherence_NLI: Uses Natural Language Inference (NLI) to assess 
               coherence based on entailment between the student's utterances and the tutor's response.
        - The Coherence_BERT metric is inspired by Dziri et al., 2019 
          (https://aclanthology.org/W19-3646/), where a higher score indicates 
          better engagement and coherence in the tutor's responses.
        - By default, the evaluation uses the Coherence_BERT metric unless specified otherwise.
        """
        super().__init__(*args, **kwargs)

        self.coherence_nli_model = checkpoint

        # Initialize NLI model and tokenizer
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.coherence_nli_model)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.coherence_nli_model)
        self.nli_model.to(self.device)

    def _calculate_nli_score(
            self, 
            convs: List[Dict[str, Any]], 
            tutor_model: str
    ) -> Union[List[float], float]:
        """
        Calculate the coherence score using NLI (natural language inference).

        Parameters:
            convs (List[Dict[str, Any]]): Conversations data.
            tutor_model (str): The name of the tutor model.

        Returns:
            List[float]: Individual NLI scores for each conversation.
            float: Average NLI score across all conversations.
        """
        all_nli_scores = []
        for example in convs:
            if utils.should_skip_example(tutor_model, example):
                continue
            studlastutt = utils.get_coherence_history(example)
            tutresp = utils.get_response_text(tutor_model, example)
            features = self.nli_tokenizer(str(studlastutt) + " [SEP] " + str(tutresp), return_tensors="pt")

            self.nli_model.eval()
            with torch.no_grad():
                scores = self.nli_model(**features.to(self.device)).logits
                softmax_score = torch.nn.functional.softmax(scores, dim=1)
                all_nli_scores.append(round(softmax_score[0][2].item(), 3))  # 'entailment' score

            utils.update_auto_annotation(tutor_model, example, round(softmax_score[0][2].item(), 3), 'Coherence_NLI')

        return all_nli_scores, sum(all_nli_scores) / len(all_nli_scores)

    def _calculate_bert_score(
            self,
            convs: List[Dict[str, Any]], 
            tutor_model: str
    ) -> Union[List[float], float]:
        """
        Calculate the coherence score using BERTScore.

        Parameters:
            convs (List[Dict[str, Any]]): Conversations data.
            tutor_model (str): The name of the tutor model.

        Returns:
            List[float]: Individual BERT scores for each conversation.
            float: Average BERT score across all conversations.
        """
        all_bert_scores = []
        for example in convs:
            if utils.should_skip_example(tutor_model, example):
                continue
            studlastutt = utils.get_coherence_history(example)
            tutresp = utils.get_response_text(tutor_model, example)
            temp_score = []
            for utt in studlastutt:
                temp_score.append(round(bertscore.compute(
                    predictions=[tutresp], references=[utt], lang="en", device=self.device, verbose=False
                )['f1'][0], 3))
            score = round(sum(temp_score) / len(temp_score), 3)
            all_bert_scores.append(score)

            utils.update_auto_annotation(tutor_model, example, score, 'Coherence_BERT')

        return all_bert_scores, sum(all_bert_scores) / len(all_bert_scores)

    def compute(
            self, 
            data: Dict[str, float] = None, 
            metrics: List[str] = None, 
            save: bool = False, 
            file_name: str = 'mrbench_coherence.json'
    ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        """
        Evaluate the coherence scores for each model using specified metrics.

        Parameters:
            data (Dict[str, float]): Input data for evaluation.
            metrics (List[str]): List of metrics to evaluate. Default is ['Coherence_BERT'].
            save (bool): Whether to save the computed results to a file.
            file_name (str): Name of the output file.

        Returns:
            Tuple[Dict[str, float], Dict[str, List[float]]]: Final scores and annotations.
        """
        if data is not None:
            self.data = data

        if metrics is None:
            metrics = ['Coherence_BERT']  # Default metric

        print(f"Computing Coherence Scores using {metrics} method(a) for {len(self.data)} examples...")

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
            'Coherence_BERT': self._calculate_bert_score,
            'Coherence_NLI': self._calculate_nli_score
        }
        if metric not in metric_methods:
            raise ValueError(f"Unsupported metric: {metric}")
        return metric_methods[metric]

    def list_available_metrics(self) -> pd.DataFrame:
        """
        List all available metrics with details.

        Returns:
            pd.DataFrame: A DataFrame containing metric names and descriptions.
        """
        metrics_details = {
            "Coherence_BERT": "Uses BERTScore to evaluate coherence between student utterance and tutor response.",
            "Coherence_NLI": "Uses Natural Language Inference (NLI) to evaluate coherence based on entailment."
        }
        return pd.DataFrame(metrics_details.items(), columns=["Metric", "Description"])

    def get_sample_examples_with_scores(
        self, 
        tutor_model: str = 'Expert', 
        num_examples: int = 5, 
        metric: str = 'Coherence_BERT'
    ) -> pd.DataFrame:

        """
        Get examples with coherence scores for a given metric and tutor model.
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
        metric: str = 'Coherence_BERT'
    ) -> pd.DataFrame:

        """
        Compare the coherence scores of two tutor models for a specific metric.

        Parameters:
            tutor_model1 (str): The name of the first tutor model.
            tutor_model2 (str): The name of the second tutor model.
            num_examples (int): The number of examples to display.
            metric (str): The metric to use for comparison.

        Returns:
            pd.DataFrame: A DataFrame with the examples and their coherence scores.
        """
        if num_examples > len(self.data):
            raise ValueError(f"Number of examples should be less than or equal to {len(self.data)}")

        metric_method = self._get_metric_method(metric)
        scores1, _ = metric_method(self.data[:num_examples], tutor_model1)
        scores2, _ = metric_method(self.data[:num_examples], tutor_model2)

        return utils.get_sample_dataframe(self.data[:num_examples], tutor_model1, scores1,
                                          tutor_model2=tutor_model2, scores2=scores2, dim=metric)