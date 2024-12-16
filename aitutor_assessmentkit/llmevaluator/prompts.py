ABS_SYSTEM_PROMPT = "You are a critic assessing a tutor who is interacting with a student by providing a clear, objective single evaluation score on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

ABSOLUTE_PROMPT = """### Task Description:
You are a critic assessing a tutor who is interacting with a student. The assessment of the ###Tutor_Response should be based on the following: ###Conversation_Topic, ###Previous_Conversation_between_Tutor_and_Student, ###Definitions_of_criteria,
###Score_Rubric, and ###Correct_Reference_Tutor Response.
1. Write a detailed feedback that assess the quality of the ###Tutor_Current_Response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer 1, 2 or 3. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer 1, 2, or 3)"
4. Please do not generate any other opening, closing, or explanations.

###Conversation_Topic: {topic}

###Previous_Conversation_between_Tutor_and_Student: {previous_conversation}

###Definitions_of_criteria: {definition}

###Score_Rubric: {rubric}

###Correct_Reference_Tutor_Response: {reference_answer}

###Tutor_Current_Response: {response}

###Feedback: """

ABSOLUTE_PROMPT_WO_REF = """### Task Description:
You are a critic assessing a tutor who is interacting with a student. The assessment of the ###Tutor_Response should be based on the following: ###Conversation_Topic, ###Previous_Conversation_between_Tutor_and_Student, ###Definitions_of_criteria, and
###Score_Rubric.
1. Write a detailed feedback that assess the quality of the ###Tutor_Current_Response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer 1, 2 or 3. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer 1, 2, or 3)"
4. Please do not generate any other opening, closing, or explanations.

###Conversation_Topic: {topic}

###Previous_Conversation_between_Tutor_and_Student: {previous_conversation}

###Definitions_of_criteria: {definition}

###Score_Rubric: {rubric}

###Tutor_Current_Response: {response}

###Feedback: """

################################
REL_SYSTEM_PROMPT = "You are a critic assessing a tutor who is interacting with a student to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."

RELATIVE_PROMPT = """### You are a critic assessing a tutor who is interacting with a student. The assessment of the ###Tutor_Response should be based on the following: ###Conversation_Topic, ###Previous_Conversation_between_Tutor_and_Student, ###Definitions_of_criteria,
###Score_Rubric, and ###Correct_Reference_Tutor Response.
1. Write detailed feedback that assesses the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing feedback, choose the better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "(write feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, or explanations.

###Conversation_Topic: {topic}

###Previous_Conversation_between_Tutor_and_Student: {previous_conversation}

###Response_A: {response_A}

###Response_B: {response_B}

###Definitions_of_criteria: {definition}

###Score_Rubric: {rubric}

###Correct_Reference_Tutor_Response: {reference_answer}

###Feedback: """

RELATIVE_PROMPT_WO_REF = """### You are a critic assessing a tutor who is interacting with a student. The assessment of the ###Tutor_Response should be based on the following: ###Conversation_Topic, ###Previous_Conversation_between_Tutor_and_Student, ###Definitions_of_criteria, and
###Score_Rubric
1. Write detailed feedback that assesses the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing feedback, choose the better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "(write feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, or explanations.

###Conversation_Topic: {topic}

###Previous_Conversation_between_Tutor_and_Student: {previous_conversation}

###Response_A: {response_A}

###Response_B: {response_B}

###Definitions_of_criteria: {definition}

###Score_Rubric: {rubric}

###Feedback: """

################################
# Definitions for each conversation dimension
DEFINITIONS = {
    "Mistake_Identification": "Mistake Identification is defined as the degree to which the tutor accurately recognizes the presence of an error in the student’s previous response.",
    "Mistake_Location": "Mistake Location is defined as the degree to which the tutor precisely identifies and specifies the exact location of the error within the student’s previous response.",
    "Revealing_of_the_Answer": "Revealing of the Answer is defined as the degree to which the tutor discloses the correct or incorrect final answer to the student.",
    "Providing_Guidance": "Providing Guidance is defined as the degree to which the tutor offers clear, accurate, and appropriate guidance, including explanations, elaborations, hints, or examples, to facilitate the student’s understanding.",
    "Coherence": "Coherence is defined as the degree to which the tutor's response demonstrates logical consistency and contextual alignment with the student’s prior responses.",
    "Actionability": "Actionability is defined as the degree to which the tutor provides clear and specific feedback outlining actionable steps for the student to address mistakes or improve understanding.",
    "Tutor_Tone": "Tutor Tone is defined as the nature of the tutor’s tone, categorized as encouraging, neutral, or offensive, and its impact on the interaction.",
    "Humanlikeness": "Humanlikeness is defined as the degree to which the tutor’s response is perceived as natural, conversational, and human-like, rather than robotic or artificial."
}

################################
# Rubrics for each conversation dimension
MISTAKE_IDENTIFICATION_RUBRIC = """
[Has the tutor identified a mistake in the student’s response?]
Score 1: The tutor fails to identify the mistake or misidentifies it.
Score 2: The tutor partially identifies the mistake but lacks precision.
Score 3: he tutor correctly identifies the mistake with high precision.
""".strip()

MISTAKE_LOCATION_RUBRIC = """
[Does the tutor’s response accurately point to a genuine mistake and its location?]
Score 1: The tutor fails to locate the mistake or mislocates it.
Score 2: The tutor partially locates the mistake but lacks precision.
Score 3: The tutor accurately locates the mistake with high precision.
""".strip()

REVEALING_ANSWER_RUBRIC = """
[Does the tutor reveal the final answer (whether correct or not)?]
Score 1: The tutor guides the student to the answer without revealing it directly.
Score 2: The tutor reveals the INCORRECT answer.
Score 3: The tutor reveals the CORRECT answer.
""".strip()

PROVIDING_GUIDANCE_RUBRIC = """
[Does the tutor offer correct and relevant guidance, such as an explanation, elaboration, hint, examples, and so on?]
Score 1: The tutor fails to provide guidance or offers misleading information.
Score 2: The tutor provides partial guidance but lacks clarity or accuracy.
Score 3: The tutor offers clear and accurate guidance to assist the student.
""".strip()

ACTIONABILITY_RUBRIC = """
[Is it clear from the tutor’s feedback what the student should do next?]
Score 1: The feedback is vague, unclear, or lacks actionable steps.
Score 2: The feedback provides some actionable steps but could be improved.
Score 3: The feedback outlines clear and actionable steps for the student.
""".strip()

COHERENCE_RUBRIC = """
[Is the tutor’s response logically consistent with the student’s previous response?]
Score 1: The response is incoherent, irrelevant, or fails to address the conversation context.
Score 2: The response is somewhat coherent and relevant, but could be improved.
Score 3: The response is coherent, relevant, and directly addresses the conversation context.
""".strip()

TUTOR_TONE_RUBRIC = """
[Is the tutor’s response encouraging, neutral, or offensive?]
Score 1: The tutor’s tone is offensive, discouraging, or inappropriate.
Score 2: The tutor’s tone is neutral but lacks encouragement.
Score 3: The tutor’s tone is encouraging, positive, and supportive.
""".strip()

HUMANLIKENESS_RUBRIC = """
[Does the tutor’s response sound natural, rather than robotic or artificial?]
Score 1: The response sounds robotic, unnatural, or machine-generated.
Score 2: The response sounds somewhat natural but lacks some human-like qualities.
Score 3: The response sounds natural, human-like, and not robotic.
""".strip()