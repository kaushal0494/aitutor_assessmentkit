import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from aitutor_assessmentkit.autoevaluator import autoeval
from aitutor_assessmentkit.helpers.constants import UPTAKE_MODEL, UPTAKE_TOKENIZER
from aitutor_assessmentkit.helpers import utils, uptake_utils


class AutoProvidingGuidanceEvaluator(autoeval.AutoEvaluator):
    """
    A class to evaluate providing guidance in tutor responses.
    """

    def __init__(self, checkpoint: str = UPTAKE_MODEL, tokenizer: str = UPTAKE_TOKENIZER, *args, **kwargs) -> None:
        """
        Initialize the AutoCoherenceEvaluator object, inheriting from AutoEvaluator.

        Parameters:
            checkpoint (str): The checkpoint for the Uptake model.
            tokenizer (str): The tokenizer associated with the Uptake model.

        Notes:
        - The providing guidance score is evaluated using the Conversational Uptake metric. For implementation, the official release by Dorottya Demszky 
          (https://github.com/ddemszky/conversational-uptake) and checkpoints from Hugging Face (https://huggingface.co/stanford-nlpxed/uptake-model) were used.
        - A higher score indicates a better performance in maintaining coherence and engaging with the student's input.
        """
        super().__init__(*args, **kwargs)

        self.uptake_checkpoint = checkpoint
        self.uptake_tokenizer_name = tokenizer

        # Initialize Uptake model and tokenizer
        self.uptake_builder, _, self.uptake_model = uptake_utils._initialize(self.uptake_checkpoint)
        self.uptake_tokenizer = AutoTokenizer.from_pretrained(self.uptake_tokenizer_name)
        self.uptake_model.to(self.device)

    def _calculate_uptake_score(
        self, 
        convs: List[Dict[str, Any]], 
        tutor_model: str
    ) -> Union[List[float], float]:
        """
        Calculate the Providing Guidance score for conversations using Conversational Uptake.

        Parameters:
            convs (List[Dict[str, Any]]): List of conversations.
            tutor_model (str): The name of the tutor model.

        Returns:
            List[float]: Individual scores for each conversation.
            float: Average score across all conversations.
        """
        all_uptake_scores = []

        for example in convs:
            if utils.should_skip_example(tutor_model, example):
                continue

            studlastutt = utils.get_providing_guidance_history(example)
            tutresp = utils.get_response_text(tutor_model, example)

            textA = uptake_utils._get_clean_text(studlastutt, remove_punct=True)
            textB = uptake_utils._get_clean_text(tutresp, remove_punct=True)

            instance = self.uptake_builder.build_inputs(
                [textB], textA, max_length=128, input_str=True
            )

            instance["attention_mask"] = [1] * len(instance["input_ids"])
            for key in ["input_ids", "token_type_ids", "attention_mask"]:
                instance[key] = torch.tensor(instance[key]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.uptake_model(
                    input_ids=instance["input_ids"],
                    attention_mask=instance["attention_mask"],
                    token_type_ids=instance["token_type_ids"],
                    return_pooler_output=False,
                )

            uptake_score = round(utils.softmax(output["nsp_logits"][0].tolist())[1], 3)
            all_uptake_scores.append(uptake_score)

            utils.update_auto_annotation(tutor_model, example, uptake_score, 'Providing_Guidance_Uptake')

        return all_uptake_scores, sum(all_uptake_scores) / len(all_uptake_scores)

    def compute(
        self, 
        data: Dict[str, Any] = None, 
        metrics: List[str] = None, 
        save: bool = False, 
        file_name: str = 'mrbench_providing_guidance.json'
    ) -> Tuple[Dict[str, float], List[float]]:
        """
        Evaluate Providing Guidance scores for each model in the dataset.

        Parameters:
            data (Dict[str, Any]): Input data for evaluation.
            metrics (List[str]): List of metrics to evaluate. Default is ['Providing_Guidance_Uptake'].
            save (bool): Whether to save the computed results to a file.
            file_name (str): Name of the output file.

        Returns:
            Tuple[Dict[str, float], List[float]]: Final scores and annotations.
        """
        if data is not None:
            self.data = data

        if metrics is None:
            metrics = ['Providing_Guidance_Uptake']  # Default metric

        print(f"Computing Providing Scores using {metrics} metric(s) for {len(self.data)} examples...")

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
            'Providing_Guidance_Uptake': self._calculate_uptake_score
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
            "Providing_Guidance_Uptake": "Providing guidance score using uptake metric."
        }
        return pd.DataFrame(methods.items(), columns=["Method", "Description"])

    def get_sample_examples_with_scores(
        self, 
        tutor_model: str = 'Expert', 
        num_examples: int = 5, 
        metric: str = 'Providing_Guidance_Uptake'
    ) -> pd.DataFrame:

        """
        Get examples with humanness scores for a given metric and tutor model.

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
        metric: str = 'Providing_Guidance_Uptake'
    ) -> pd.DataFrame:

        """
        Compare the humanness scores of two tutor models for a specific metric.

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
