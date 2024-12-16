# """
# Computation of Intensity, Valence, Arousal.
# The implementation is inspired from https://github.com/MichiganNLP/empathy_eval
# """

import os
import json
import pickle
import math
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize

import time
import pandas as pd
import re
from pathlib import Path
import datetime;
from aitutor_assessmentkit.helpers.constants import VAD_PATH

# p = VAD_PATH
# def load_intensity():
#     """
#     Load intensity info.
#     """
#     with open(os.path.join(p, "NRC-Emotion-Intensity-Lexicon-v1.txt"), "r") as file_p:
#         intensity_lex = file_p.readlines()  #outraged	anger	0.964

#     intensities = {}
#     i = 0
#     for intensity in intensity_lex:
#         tokens = intensity.split("	")
#         intensities[tokens[0]] = float(tokens[2])
#         # if(i<10):
#         #     print(tokens)
#         #     i += 1
#     return intensities


# def load_vad():
#     """
#     Load intensity info.
#     """
#     with open(os.path.join(p, "NRC-VAD-Lexicon.txt"), "r") as file_p:
#         vad_lex = file_p.readlines()  #aaaaaaah	0.479	0.606	0.291

#     vad = {}
#     i = 0
#     for vad_ in vad_lex:
#         tokens = vad_.split("	")
#         # if(i<10):
#         #     print(tokens)
#         #     i += 1

#         vad[tokens[0]] = [float(tokens[1]),float(tokens[2]),float(tokens[3])]
#     return vad



# VAD = load_vad()
# INTENSITIES = load_intensity()

# def get_token_intensity(token):
#     return INTENSITIES.get(token, 0)

# def get_token_vad(token):
#     """
#     Get VAD vector
#     """
#     return VAD.get(token, [0, 0, 0])

# def get_intensity(query):
#     tokens = word_tokenize(query.lower())
#     #tmp = [get_token_intensity(token) for token in tokens]
#     #tmp2 = [token if i>0 for i in tmp]
#     return [get_token_intensity(token) for token in tokens]

# def get_intensity_token(query):
#     tokens = word_tokenize(query.lower())
#     tmp = [get_token_intensity(token) for token in tokens]
#     tmp2 = [token for i,token in zip(tmp,tokens) if i>0]
#     return tmp2


# def get_vad(query):
#     """
#     Get mean, max scores for VAD.
#     """
#     tokens = word_tokenize(query.lower())
#     vads = [get_token_vad(token) for token in tokens]
#     vads = [x for x in vads if x is not None]

#     valence = [x[0] for x in vads]
#     arousal = [x[1] for x in vads]
#     dominance = [x[2] for x in vads]
#     return valence, arousal, dominance


# #def get_vad_stats(data, system):
# def get_vad_stats(last_responses, generation, system):
#     """
#     Compute intensity, vad.
#     """

#     results = []

#     #for convo_obj in data:
#     for last_utt, response in zip(last_responses,generation):
#         #context = convo_obj["query"]
#         #last_utt = context[-1]
#         #response = convo_obj["response"]

#         context_v, context_a, context_d = get_vad(last_utt)
#         #print(context_v, context_a, context_d)
#         response_v, response_a, response_d = get_vad(response)

#         context_intensity = get_intensity(last_utt)
#         response_intensity = get_intensity(response)
#         # print(context_intensity,last_utt)
#         # print("_______________")
#         # print(response_intensity,response)
#         # print("_______________")


#         max_v_context = 0
#         max_a_context = 0
#         max_d_context = 0
#         mean_v_context = 0
#         mean_a_context = 0
#         mean_d_context = 0

#         if len(context_v) > 0:
#             max_v_context = max(context_v)
#             mean_v_context = np.mean(context_v)
#         if len(context_a) > 0:
#             max_a_context = max(context_a)
#             mean_a_context = np.mean(context_a)
#         if len(context_d) > 0:
#             max_d_context = max(context_d)
#             mean_d_context = np.mean(context_d)

#         if len(response_v) > 0:
#             max_v = max(response_v)
#             mean_v = np.mean(response_v)
#         if len(response_a) > 0:
#             max_a = max(response_a)
#             mean_a = np.mean(response_a)
#         if len(response_d) > 0:
#             max_d = max(response_d)
#             mean_d = np.mean(response_d)

#         diff_max_v = max_v_context - max_v
#         diff_mean_v = mean_v_context - mean_v
#         diff_max_a = max_a_context - max_a
#         diff_mean_a = mean_a_context - mean_a
#         diff_max_d = max_d_context - max_d
#         diff_mean_d = mean_d_context - mean_d
#         diff_intensity = max(context_intensity) - max(response_intensity)

#         results.append(
#             {
#                 "max_v": max_v,
#                 "mean_v": mean_v,
#                 "max_a": max_a,
#                 "mean_a": mean_a,
#                 "max_d": max_d,
#                 "mean_d": mean_d,
#                 "diff_max_v": diff_max_v,
#                 "diff_mean_v": diff_mean_v,
#                 "diff_max_a": diff_max_a,
#                 "diff_mean_a": diff_mean_a,
#                 "diff_max_d": diff_max_d,
#                 "diff_mean_d": diff_mean_d,
#                 "diff_max_intensity": diff_intensity,
#             }
#         )

#     return results

# def compare_vad(filepaths , c=0):
#     """ Compare VADs """
#     # To get last responses
#     ds = np.load("datasets/sys_dialog_texts.test.npy", allow_pickle=True) #UPDATE: last user dialog path
#     last_dialog = [l[-1] for l in ds] # UPDATE to get a list of last user dialogs

#     scores = {}
#     for system, filepath in filepaths:
#         print(filepath)
#         df = pd.read_json(filepath, lines=True) # reading generations from json file UPDATE as needed
#         col = ['prompt0','prompt1'] # Columns containing the  generations
#         c_name = col[c]
#         responses = list(df[c_name]) # UPDATE to get a list of generations

#         print(f"Len of responses: {len(responses)} , len data: {len(ds)}")

#         vad_stats = get_vad_stats(last_dialog, responses, system) #get_vad_stats(data, system)

#         diff_max_v = np.mean([x["diff_max_v"] for x in vad_stats])
#         diff_max_a = np.mean([x["diff_max_a"] for x in vad_stats])
#         diff_max_d = np.mean([x["diff_max_d"] for x in vad_stats])
#         diff_max_intensity = np.mean(
#             [x["diff_max_intensity"] for x in vad_stats]
#         )

#         with open(out_file, 'a+') as fo:
#             fo.write("--\n")
#             fo.write(f"-{c}-\n")
#             fo.write("--\n")
#             fo.write(f"({system}) Diff Max V: {diff_max_v}\n")
#             fo.write(f"({system}) Diff Max A: {diff_max_a}\n")
#             fo.write(f"({system}) Diff Max D: {diff_max_d}\n")
#             fo.write(f"({system}) Diff Intensity: {diff_max_intensity}\n")

#         scores[system] = {
#             "diff_max_v": diff_max_v,
#             "diff_max_a": diff_max_a,
#             "diff_max_d": diff_max_d,
#             "diff_max_intensity": diff_max_intensity,
#         }
#     return scores



def main():
    """ Driver """

    # To get first generation
    gen_path = "generations/empchat/ver2/" # UPDATE  - directory with generations

    # UPDATE "systems" - Add list of generation files
    systems = [#"0shot/falcon_instruct_7b_cleaned", example
               ]
    
    filepaths = [(system, os.path.join(gen_path, f"{system}.json")) for system in systems]
    compare_vad(filepaths)
    compare_vad(filepaths,1)



if __name__ == "__main__":
    
    start = time.time()
    # ct stores current time
    ct = datetime.datetime.now()
    pname = "evaluations/empchat/ver2/" # UPDATE: Path to save the outputs
    path = Path(pname)
    path.mkdir(parents=True, exist_ok=True)
    out_file = pname +"vad.txt"
    print("OUTfile: ", out_file)
    with open(out_file, 'a+') as fo:
        fo.write(f"_________________________{ct}\n")

    main()

    end = time.time()
    ep = end - start
    with open(out_file, 'a+') as fo:
        fo.write(f"Total time elapsed (seconds): {ep}\n")
        fo.write(f"Total time elapsed (minutes): {ep/60}\n")
        fo.write(f"Total time elapsed (hours): {ep/(60*60)}\n")
        fo.write("__________________________________________________________________\n")