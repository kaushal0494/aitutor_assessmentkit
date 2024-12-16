VALID_FILE_EXTENSIONS = [".json", ".csv"]
COMMON_COLUMNS = ['conversation_id', 'conversation_history', 'anno_llm_responses']

# Annotation labels encoding
LABELS_ENCODING = {'Mistake_Identification': {'Yes': 1, 'To some extent': 0.5, 'No': 0},
                   'Mistake_Location': {'Yes': 1, 'To some extent': 0.5, 'No': 0},
                   'Revealing_of_the_Answer': {'Yes (and the answer is correct)': 1, 'Yes (but the answer is incorrect)': 0.5, 'No': 0},
                   'Providing_Guidance': {'Yes': 1, 'To some extent': 0.5, 'No': 0},
                   'Actionability': {'Yes': 1, 'To some extent': 0.5, 'No': 0},
                   'Coherence': {'Yes': 1, 'To some extent': 0.5, 'No': 0},
                   'Tutor_Tone': {'Encouraging': 1, 'Neutral': 0.5, 'Offensive': 0},
                   'Humanlikeness': {'Yes': 1, 'To some extent': 0.5, 'No': 0},}

LLM_LABELS_ENCODING = {3: 1, 2: 0.5, 1: 0}

# LABELS_ENCODING = {'Mistake_Identification': {'Yes': 3, 'To some extent': 2, 'No': 1},
#                    'Mistake_Location': {'Yes': 3, 'To some extent': 2, 'No': 1},
#                    'Revealing_of_the_Answer': {'Yes (and the answer is correct)': 3, 'Yes (but the answer is incorrect)': 2, 'No': 1},
#                    'Providing_Guidance': {'Yes': 3, 'To some extent': 2, 'No': 1},
#                    'Actionability': {'Yes': 3, 'To some extent': 2, 'No': 1},
#                    'Coherence': {'Yes': 3, 'To some extent': 2, 'No': 1},
#                    'Tutor_Tone': {'Encouraging': 3, 'Neutral': 2, 'Offensive': 1},
#                    'Humanlikeness': {'Yes': 3, 'To some extent': 2, 'No': 1},}

# Best models for each metrics
BEST_AUTO_MODELS = {
    'Mistake_Identification': ['Mistake_Identification_Heuristic'],
    'Mistake_Location': ['Mistake_Location_Heuristic'],
    'Revealing_of_the_Answer': ['Revealing_of_the_Answer_Heuristic'],
    'Providing_Guidance':['Providing_Guidance_Uptake'],
    'Actionability': ['Actionability_Heuristic'],
    'Coherence': ['Coherence_BERT'],
    'Tutor_Tone': ['Tutor_Tone_FTRoBERTa'],
    'Humanlikeness': ['Humanlikeness_OGPT2'],
}

# Best models for each metrics
ALL_AUTO_MODELS = {
    'Mistake_Identification': ['Mistake_Identification_Heuristic'],
    'Mistake_Location': ['Mistake_Location_Heuristic'],
    'Revealing_of_the_Answer': ['Revealing_of_the_Answer_Heuristic'],
    'Providing_Guidance':['Providing_Guidance_Uptake'],
    'Actionability': ['Actionability_Heuristic'],
    'Coherence': ['Coherence_BERT', 'Coherence_NLI'],
    'Tutor_Tone': ['Tutor_Tone_FTRoBERTa'],
    'Humanlikeness': ['Humanlikeness_OGPT2', 'Humanlikeness_Heuristic'],
}

BEST_LLM_MODELS = {
    'Mistake_Identification': ['Mistake_Identification_prometheus-eval/prometheus-7b-v2.0'],
    'Mistake_Location': ['Mistake_Location_prometheus-eval/prometheus-7b-v2.0'],
    'Revealing_of_the_Answer': ['Revealing_of_the_Answer_prometheus-eval/prometheus-7b-v2.0'],
    'Providing_Guidance':['Providing_Guidance_prometheus-eval/prometheus-7b-v2.0'],
    'Actionability': ['Actionability_prometheus-eval/prometheus-7b-v2.0'],
    'Coherence': ['Coherence_prometheus-eval/prometheus-7b-v2.0'],
    'Tutor_Tone': ['Tutor_Tone_prometheus-eval/prometheus-7b-v2.0'],
    'Humanlikeness': ['Humanlikeness_prometheus-eval/prometheus-7b-v2.0'],
}

# BEST_LLM_MODELS = {
#     'Mistake_Identification': ['Mistake_Identification_meta-llama/Meta-Llama-3.1-8B-Instruct'],
#     'Mistake_Location': ['Mistake_Location_meta-llama/Meta-Llama-3.1-8B-Instruct'],
#     'Revealing_of_the_Answer': ['Revealing_of_the_Answer_meta-llama/Meta-Llama-3.1-8B-Instruct'],
#     'Providing_Guidance':['Providing_Guidance_meta-llama/Meta-Llama-3.1-8B-Instruct'],
#     'Actionability': ['Actionability_meta-llama/Meta-Llama-3.1-8B-Instruct'],
#     'Coherence': ['Coherence_meta-llama/Meta-Llama-3.1-8B-Instruct'],
#     'Tutor_Tone': ['Tutor_Tone_meta-llama/Meta-Llama-3.1-8B-Instruct'],
#     'Humanlikeness': ['Humanlikeness_meta-llama/Meta-Llama-3.1-8B-Instruct'],
# }

# Public trained models
GRAMMATICALITY_PUBLIC_COLA_MODEL="textattack/roberta-base-CoLA"
UPTAKE_MODEL="ddemszky/uptake-model"
UPTAKE_TOKENIZER="bert-base-uncased"
HUMANLIKENESS_OPENAI_ROBERA_MODEL="openai-community/roberta-large-openai-detector"
COHERENCE_NLI_MODEL="cross-encoder/nli-deberta-v3-large"

# Our own models
PEDAGOGY_STORY_PATH="/fsx/hyperpod-output-artifacts/AROA6GBMFKRI2VWQAUGYI:Kaushal.Maurya@mbzuai.ac.ae/PAA/outputs/story_peda_classifier/exp_01"
TUTORTONE_MODEL="/home/kaushal.maurya/AITutor_AssessmentKit/resources/tuntortone_ckeckpoint/exp_01"


# Resources
VAD_PATH="/fsx/homes/Kaushal.Maurya@mbzuai.ac.ae/PAA/resource/"
RESOURCE_PATH="/fsx/homes/Kaushal.Maurya@mbzuai.ac.ae/PAA/resource/"
ED_COUNT_PATH="/fsx/hyperpod-input-datasets/AROA6GBMFKRI2VWQAUGYI:Kaushal.Maurya@mbzuai.ac.ae/PAA/resource/bridge_dict.pkl"