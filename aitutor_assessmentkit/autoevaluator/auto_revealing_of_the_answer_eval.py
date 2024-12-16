import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from tqdm import tqdm

from aitutor_assessmentkit.autoevaluator import autoeval
from aitutor_assessmentkit.helpers import utils


class AutoRevealingOfTheAnswerEvaluator(autoeval.AutoEvaluator):
    """
    Evaluates the ability of tutor models to reveal the final answer to the student.
       
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the AutoRevealingoftheAnswerEvaluator.

        Notes:

        - **Keyword-Based Evaluation:**
          - Focuses on linguistic cues and patterns often associated with Revealing of the Answer.
            - Uses a heuristic-based approach to evaluate the Revealing of the Answer ability of tutor models.
            - Lower scores the tutor is better by not revealing the final answer immediately.
        """
        super().__init__(*args, **kwargs)

    def _calculate_heuristic_based_score(self, convs: List[Dict[str, Any]], tutor_model: str) -> Union[List[float], float]:
        """
        Compute revealing of the answer scores using heuristics.

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

            keywords = ["remember", "think", "actually", "mistake", "cost", "times"]
            response_lower = response.lower()
            if any(keyword in response_lower for keyword in keywords):
                score = 1.0
            else:
                score = 0.0

            scores.append(score)
            utils.update_auto_annotation(tutor_model, example, score, 'Revealing_of_the_Answer_Heuristic')
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
            'Revealing_of_the_Answer_Heuristic': self._calculate_heuristic_based_score
        }
        if metric not in metric_methods:
            raise ValueError(f"Unsupported metric: {metric}")
        return metric_methods[metric]
    
    def compute(
        self, 
        data: Dict[str, Any] = None, 
        metrics: List[str] = None,
        save: bool = False, 
        file_name: str = 'mrbench_revealing_of_the_answer.json'
    ) -> Tuple[Dict[str, Any], List[float]]:
        """
        Compute revealing of the answer scores for tutor models.

        Parameters:
            data (Dict[str, Any]): Input data for evaluation.
            method (List[str]): Evaluation mwtrics to use. Default is ['Mistake_Identification_Heuristic'].
            save (bool): Whether to save the results.
            file_name (str): File name for saving results.

        Returns:
            Dict[str, Any]: Final scores.
            List[float]: Collected annotations.
        """
        if data is not None:
            self.data = data

        if metrics is None:
            metrics = ['Revealing_of_the_Answer_Heuristic']

        print(f"Computing Revealing of the_Answer Scores using {metrics} mtrics(s) for {len(self.data)} examples...")

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
            "Revealing_of_the_Answer_Heuristic": "Compute revealing of the answer scores using heuristics."
        }
        return pd.DataFrame(methods.items(), columns=["Method", "Description"])

    def get_sample_examples_with_scores(
        self, 
        tutor_model: str = 'Expert', 
        num_examples: int = 5, 
        metric: str = 'Revealing_of_the_Answer_Heuristic',
    ) -> pd.DataFrame:

        """
        Get examples with revealing of the answer scores for a specific tutor model.

        Parameters:
            tutor_model (str): The name of the tutor model.
            num_examples (int): The number of examples to display.
            metric (str): The metric to use for evaluation.

        Returns:
            pd.DataFrame: A DataFrame with the examples and their humanness scores.
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
        metric: str = 'Revealing_of_the_Answer_Heuristic'
    ) -> pd.DataFrame:

        """
        Compare the mistake location scores of two tutor models.

        Parameters:
            tutor_model1 (str): The name of the first tutor model.
            tutor_model2 (str): The name of the second tutor model.
            num_examples (int): The number of examples to display.
            metric (str): The metric to use for comparison.

        Returns:
            pd.DataFrame: A DataFrame with the examples and their humanness scores.
        """
        if num_examples > len(self.data):
            raise ValueError(f"Number of examples should be less than or equal to {len(self.data)}")

        metric_method = self._get_metric_method(metric)
        scores1, _ = metric_method(self.data[:num_examples], tutor_model1)
        scores2, _ = metric_method(self.data[:num_examples], tutor_model2)

        return utils.get_sample_dataframe(self.data[:num_examples], tutor_model1, scores1,
                                          tutor_model2=tutor_model2, scores2=scores2, dim=metric)