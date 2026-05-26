"""
agents.py
---------
This module defines the agent classes: PatientAgent, DoctorAgent, and MeasurementAgent.
Each agent wraps its own conversation history, system prompt, and uses the shared query_model
function to obtain responses from various LLM backends.
"""

import time
import re
import random
from medsim.query_model import (
    query_model,
    BAgent,
    extract_question,
    generate_possible_diagnoses,
    generate_possible_diagnoses_from_discussion,
    generate_answer_choices,
    generate_answer_choices_from_candidates,
    generate_question_json,
    generate_question_json_no_answer,
    import_generate,
    get_diagnosis,
)
# from query import BAgent
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("agents")
class PatientAgent:
    """Simulates a patient in the dialogue simulation."""

    def __init__(self, scenario=None, backend_str="gpt4", bias_present=None, prompt_version="optimized"):
        self.disease = ""  # The underlying disease (the correct answer)
        self.symptoms = ""  # Symptoms to present in dialogue
        self.agent_hist = ""
        self.backend = backend_str
        self.bias_present = None if bias_present == "None" else bias_present
        self.scenario = scenario
        self.prompt_version = prompt_version
        if scenario is not None:
            self.reset()

    def update_scenario(self, scenario, bias_present=None):
        self.scenario = scenario
        if bias_present is not None:
            self.bias_present = None if bias_present == "None" else bias_present
        self.reset()

        # List of supported biases (for reference)
        self.biases = [
            "recency", "frequency", "false_consensus", "self_diagnosis",
            "gender", "race", "sexual_orientation", "cultural", "education",
            "religion", "socioeconomic"
        ]

    def generate_bias(self) -> str:
        """Generates bias-specific text to be appended to the system prompt."""
        if self.bias_present == "recency":
            return ("\nRecently, you had a friend with similar symptoms that was diagnosed with cancer. "
                    "This affects how you interact with the doctor.\n")
        elif self.bias_present == "frequency":
            return ("\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. "
                    "This affects how you interact with the doctor.\n")
        elif self.bias_present == "false_consensus":
            return ("\nMost of your friends believe that you have cancer. "
                    "This affects how you interact with the doctor.\n")
        elif self.bias_present == "self_diagnosis":
            return ("\nAfter researching your symptoms online, you are convinced that you have cancer. "
                    "This affects how you interact with the doctor.\n")
        elif self.bias_present in ["gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]:
            return (f"\nYou are uncomfortable with your doctor because of their {self.bias_present} background, "
                    "and you do not trust their judgement. This affects your responses.\n")
        else:
            # If no supported bias is specified, do nothing.
            return ""

    def system_prompt(self) -> str:
        """Builds the system prompt for the patient agent."""
        bias_prompt = self.generate_bias() if self.bias_present is not None else ""
        if self.prompt_version == "original":
            base = (
                "You are a patient in a clinic who only responds in the form of dialogue. "
                "You are being inspected by a doctor who will ask you questions and perform exams "
                "to determine your disease. Your answer will only be 1-3 sentences long.\n\n"
                f"Below is all your information:\n{self.symptoms}\n\n"
                "Remember, you must not reveal your disease explicitly but may only describe your symptoms."
            )
            return base + bias_prompt
        else:
            base = (
                "You are a patient visiting a doctor. Respond naturally in dialogue.\n\n"
                "RULES:\n"
                "1. Describe your symptoms clearly and specifically.\n"
                "2. Include relevant details: when symptoms started, how severe they are, what triggers or worsens them.\n"
                "3. If asked about your medical history, family history, or lifestyle, answer honestly with details from your information.\n"
                "4. Keep responses to 1-3 sentences.\n"
                "5. NEVER reveal your disease name. Only describe what you experience.\n"
                "6. Be cooperative. Answer the doctor's questions directly and helpfully."
            )
            symptoms = (
                f"\n\nYour medical information:\n{self.symptoms}\n\n"
                "Use ALL details from this information when answering the doctor. "
                "Do not hold back relevant clinical details."
            )
            return base + bias_prompt + symptoms

    def inference_patient(self, doctor_question: str) -> str:
        """Queries the language model for the patient's next dialogue turn."""
        if self.prompt_version == "original":
            prompt = (
                "Here is a history of your dialogue: " + self.agent_hist +
                "\nHere was the doctor response: " + doctor_question +
                "\nNow please continue your dialogue\n"
                "Patient: "
            )
        else:
            prompt = (
                "=== DIALOGUE HISTORY ===\n" + self.agent_hist +
                "\n=== DOCTOR ASKED ===\n" + doctor_question +
                "\n=== YOUR TASK ===\n"
                "Answer the doctor's question using your medical information. "
                "Be specific and include relevant clinical details. "
                "Do NOT name your disease.\n"
                "Patient: "
            )
        answer = query_model(self.backend, prompt, self.system_prompt())
        if answer is None:
            answer = "I'm not sure how to answer that."
        self.agent_hist += doctor_question + "\n\n" + answer + "\n\n"
        return answer

    def add_hist(self, hist_str: str) -> None:
        """Adds additional history to the dialogue."""
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        """Resets the patient agent state."""
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

from typing import Optional, List, Dict, Any

# class DoctorAgent:
#     """Simulates a doctor in a medical dialogue scenario with configurable biases and collaborative diagnosis."""

#     BIAS_PROMPTS: Dict[str, str] = {
#         "recency": (
#             "\nRecently, you learned that a doctor with similar patient symptoms diagnosed cancer. "
#             "This affects how you interact with the patient.\n"
#         ),
#         "frequency": (
#             "\nYou recently learned that most frequently, doctors diagnose such symptoms with cancer. "
#             "This affects your approach.\n"
#         ),
#         "false_consensus": (
#             "\nMost of your colleagues believe patients with these symptoms have cancer. "
#             "This influences your interaction.\n"
#         ),
#         "status_quo": (
#             "\nYour hospital has diagnosed these symptoms as cancer for 10 years. "
#             "This affects your approach.\n"
#         ),
#         "confirmation": (
#             "\nYou are initially confident the patient has cancer. "
#             "This affects your questioning.\n"
#         )
#     }
#     SOCIAL_BIASES: List[str] = [
#         "gender", "race", "sexual_orientation",
#         "cultural", "education", "religion", "socioeconomic"
#     ]

#     def __init__(
#         self,
#         scenario: Any,
#         backend_str: str = "gpt4",
#         max_infs: int = 20,
#         bias_present: Optional[str] = None,
#         img_request: bool = False,
#         num_doctors: int = 3,
#         debate_rounds: int = 2
#     ) -> None:
#         if max_infs < 1:
#             raise ValueError("max_infs must be at least 1")

#         self.scenario = scenario
#         self.backend = backend_str
#         self.MAX_INFS = max_infs
#         self.bias_present = None if bias_present in (None, "None") else bias_present
#         self.img_request = img_request
#         self.num_doctors = num_doctors
#         self.debate_rounds = debate_rounds
#         self.infs = 0
#         self.agent_hist = ""
#         self.presentation = scenario.examiner_information()
#         self.reset()

#     def reset(self) -> None:
#         """Resets conversation history while retaining initial setup."""
#         self.agent_hist = ""
#         self.presentation = self.scenario.examiner_information()
#         self.infs = 0

#     def generate_bias(self) -> str:
#         """Generates contextual bias prompt if configured."""
#         if self.bias_present in self.BIAS_PROMPTS:
#             return self.BIAS_PROMPTS[self.bias_present]
#         elif self.bias_present in self.SOCIAL_BIASES:
#             return (
#                 f"\nYou feel uncomfortable with the patient's {self.bias_present} background, "
#                 "affecting your trust in their judgment.\n"
#             )
#         return ""

#     def system_prompt(self) -> str:
#         """Constructs the core prompt with dynamic elements."""
#         base = [
#             "You are Dr. Agent, a medical professional conducting patient interviews.",
#             "Ask targeted questions to explore differential diagnoses.",
#             f"Max questions: {self.MAX_INFS} (used {self.infs}).",
#             "In final round, collaborate with other experts for consensus diagnosis.",
#             "Format: 'REQUEST TEST: [test]' for tests. Keep responses concise (1-3 sentences).",
#             "Final diagnosis format: 'DIAGNOSIS READY: [diagnosis]'."
#         ]
#         if self.img_request:
#             base.append("Use 'REQUEST IMAGES' for medical imaging.")

#         return (
#             " ".join(base)
#             + self.generate_bias()
#             + f"\n\nClinical Context: {self.presentation}"
#         )

#     def inference_doctor(self, patient_response: str, image_requested: bool = False) -> str:
#         """Handles a single turn of doctor-patient interaction."""
#         if self.infs >= self.MAX_INFS:
#             return "Maximum inference limit reached."

#         # Final round triggers multi-doctor debate
#         if self.infs == self.MAX_INFS - 1:
#             return self._conduct_consensus_discussion(patient_response, image_requested)

#         response = self._query_llm(
#             f"Dialogue History: {self.agent_hist}\nPatient Response: {patient_response}\nDoctor: ",
#             image_requested
#         )
#         self._update_conversation(patient_response, response)
#         return response

#     def _conduct_consensus_discussion(self, patient_stmt: str, image_requested: bool) -> str:
#         """Facilitates multi-round debate between medical experts."""
#         debate_history = self._initial_opinions_round(patient_stmt, image_requested)
#         for _ in range(self.debate_rounds - 1):
#             debate_history = self._rebuttal_round(debate_history, image_requested)

#         return self._finalize_diagnosis(debate_history)

#     def _initial_opinions_round(self, patient_stmt: str, image_requested: bool) -> Dict[str, str]:
#         """Gathers initial expert opinions."""
#         context = f"Patient Statement: {patient_stmt}\nHistory Summary: {self._summarize_history()}"
#         roles = [f"Specialist {i+1}" for i in range(self.num_doctors)]

#         return {
#             role: self._query_llm(
#                 f"{context}\n{role}, provide:\n1. Top diagnoses\n2. Key findings\n3. Confidence\n4. Missing data\n{role}: ",
#                 image_requested,
#                 collaborative=True
#             )
#             for role in roles
#         }

#     def _rebuttal_round(self, previous_opinions: Dict[str, str], image_requested: bool) -> Dict[str, str]:
#         """Performs a round of critical analysis of existing opinions."""
#         updated = {}
#         for role, opinion in previous_opinions.items():
#             others = "\n".join(f"{r}: {o}" for r, o in previous_opinions.items() if r != role)
#             updated[role] = self._query_llm(
#                 f"Current Opinions:\n{others}\n{role}, analyze weaknesses and refine diagnosis:",
#                 image_requested,
#                 collaborative=True
#             )
#         return updated

#     def _finalize_diagnosis(self, debate_history: Dict[str, str]) -> str:
#         """Generates final consensus diagnosis from expert discussion."""
#         consolidated = "\n".join(f"{r}:\n{o}" for r, o in debate_history.items())
#         return self._query_llm(
#             f"Reach consensus diagnosis from:\n{consolidated}\nFinal Diagnosis:",
#             collaborative=True
#         ).replace("DIAGNOSIS READY:", "").strip()

#     def _query_llm(self, prompt: str, image_requested: bool=None, collaborative: bool = False) -> str:
#         """Centralized LLM query handler with error checking."""
#         try:
#             return query_model(
#                 self.backend,
#                 prompt,
#                 system_prompt=self.col_system_prompt() if collaborative else self.system_prompt(),
#                 image_requested=image_requested,
#                 scene=self.scenario
#             )
#         except Exception as e:
#             return f"Error in model query: {str(e)}"

#     def _update_conversation(self, patient_input: str, doctor_response: str) -> None:
#         """Maintains conversation history with truncation for efficiency."""
#         self.agent_hist += f"Patient: {patient_input}\nDoctor: {doctor_response}\n"
#         self.infs += 1
#         # Keep last 2000 characters to prevent memory bloat
#         if len(self.agent_hist) > 2000:
#             self.agent_hist = "..." + self.agent_hist[-1500:]

#     def _summarize_history(self) -> str:
#         """Smart summarization focusing on medical relevance."""
#         return "\n".join([
#             line for line in self.agent_hist.split("\n")
#             if any(key in line.lower() for key in ["symptom", "test", "pain", "history"])
#         ][-5:])  # Keep last 5 relevant lines

#     def col_system_prompt(self) -> str:
#         """Collaboration-specific instructions for expert panel."""
#         return (
#             "You are a medical expert in a diagnostic council. "
#             "Analyze findings rigorously, challenge assumptions, and aim for consensus. "
#             "Format responses as 'Role: [Content]'. Conclude with unified DIAGNOSIS READY."
#         )
class DoctorAgent:
    """Simulates a doctor in the dialogue simulation."""

    def __init__(self, scenario=None, backend_str="gpt4", max_infs=20, bias_present=None, img_request=False, prompt_version="optimized"):
        self.infs = 0  # Inference call counter
        self.MAX_INFS = max_infs
        self.agent_hist = ""
        self.presentation = ""
        self.backend = backend_str
        self.bias_present = None if bias_present == "None" else bias_present
        self.scenario = scenario
        self.img_request = img_request
        self.num_doctors = 5
        self.prompt_version = prompt_version
        if scenario is not None:
            self.reset()

    def update_scenario(self, scenario, bias_present=None, max_infs=None, img_request=None):
        self.scenario = scenario
        if bias_present is not None:
            self.bias_present = None if bias_present == "None" else bias_present
        if max_infs is not None:
            self.MAX_INFS = max_infs
        if img_request is not None:
            self.img_request = img_request
        self.infs = 0
        self.agent_hist = ""
        self.reset()

        self.biases = [
            "recency", "frequency", "false_consensus", "confirmation", "status_quo",
            "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"
        ]

    def generate_bias(self) -> str:
        """Generates bias text for the doctor agent."""
        if self.bias_present == "recency":
            return ("\nRecently, you learned that a doctor with similar patient symptoms diagnosed cancer. "
                    "This affects how you interact with the patient.\n")
        elif self.bias_present == "frequency":
            return ("\nYou recently learned that most frequently, doctors diagnose such symptoms with cancer. "
                    "This affects your approach.\n")
        elif self.bias_present == "false_consensus":
            return ("\nMost of your colleagues believe that patients with these symptoms have cancer. "
                    "This influences your interaction.\n")
        elif self.bias_present == "status_quo":
            return ("\nYour hospital has been diagnosing these symptoms with cancer for the past 10 years. "
                    "This affects your approach.\n")
        elif self.bias_present == "confirmation":
            return ("\nYou are initially confident that the patient has cancer. "
                    "This affects your questioning.\n")
        elif self.bias_present in ["gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]:
            return (f"\nYou are uncomfortable with your patient because of their {self.bias_present} background, "
                    "and you do not trust their judgement. This affects your interaction.\n")
        else:
            return ""

    def system_prompt(self) -> str:
        """Builds the system prompt for the doctor agent with instructions for both normal and final inference rounds."""
        bias_prompt = self.generate_bias() if self.bias_present is not None else ""
        if self.prompt_version == "original":
            base = (
                "You are a doctor named Dr. Agent who only responds in the form of dialogue. "
                "In your normal rounds, your goal is to thoroughly explore possible diseases by asking diverse, "
                "targeted questions. Ask about the patient's symptoms, medical history, and any other details "
                "that might help you narrow down the potential diagnoses. "
                f"You are allowed a maximum of {self.MAX_INFS} questions before you must arrive at a diagnosis, "
                f"and you have asked {self.infs} questions so far. "
                "When you are in the final inference round, your diagnosis will not come from your individual "
                "assessment alone. Instead, a internal discussion among experts will be used to decide the final "
                "disease prediction. You can request test results using the format 'REQUEST TEST: [test]'. "
                "For example, 'REQUEST TEST: Chest_X-Ray'. Your dialogue responses should be concise "
                "(1-3 sentences long). Once you are ready with your diagnosis, please indicate it by typing "
                "'DIAGNOSIS READY: [diagnosis here]'."
            )
            if self.img_request:
                base += " You may also request medical images using 'REQUEST IMAGES'.\n"
            presentation = f"\n\n{self.presentation}"
            return base + bias_prompt + presentation
        else:
            base = (
                "You are Dr. Agent, a physician conducting a diagnostic interview.\n\n"

                "GOAL: Diagnose the patient's disease through systematic questioning and targeted tests.\n"
                f"LIMIT: You have {self.MAX_INFS} total turns. You have used {self.infs} so far.\n\n"

                "RULES:\n"
                "1. NEVER repeat a question or test you already asked.\n"
                "2. Each turn, ask ONE new question about a topic you have NOT covered yet.\n"
                "3. NEVER request a test you already ordered.\n"
                "4. When ready, output ONLY: DIAGNOSIS READY: [single disease name]\n"
                "5. Do NOT give multiple diagnoses or say 'or'. Pick the ONE most likely disease.\n"
                "6. Do NOT say 'need further testing' in your diagnosis.\n\n"

                "TEST ORDERING:\n"
                "- Use format: REQUEST TEST: [test name]\n"
                "- Example: REQUEST TEST: Complete Blood Count\n"
                "- Order basic tests first (blood work, urinalysis), then advanced tests (CT, MRI, EMG).\n\n"

                "DIAGNOSTIC STRATEGY:\n"
                "Phase 1 (turns 1-4): Ask about onset, duration, severity, triggers, past medical history, family history.\n"
                "Phase 2 (turns 5-10): Request targeted tests based on symptoms. Order each test ONLY ONCE.\n"
                "Phase 3 (turns 11+): If still uncertain, ask about specific differentiating features.\n\n"

                "Keep each response to 1-3 sentences. You may combine one question with one test request per turn."
            )
            if self.img_request:
                base += " You may also request medical images using 'REQUEST IMAGES'.\n"
            presentation = (
                f"\n\nClinical Context: {self.presentation}\n\n"
                "Use this information to ask relevant questions and gather necessary details."
            )
            return base + bias_prompt + presentation

    def inference_doctor(self, patient_response: str, image_requested: bool = False, scenario_id=0) -> str:
        """
        Queries the language model for the doctor’s next dialogue turn.
        If this is the final inference round (self.infs == self.MAX_INFS - 1),
        conduct a multi-agent, multi-round debate for the final disease prediction.
        """
        if self.infs >= self.MAX_INFS:
            return "Maximum inferences reached"

        # Final round: no longer triggers internal_discussion here;
        # debate verification is handled by run.py after diagnosis is proposed.

        # Detect final round (run.py passes "This is the final question..." as patient_response)
        is_final_round = "This is the final question" in patient_response

        # Otherwise, proceed with the normal single-agent dialogue turn.
        if is_final_round:
            prompt = (
                "=== DIALOGUE HISTORY ===\n" + self.agent_hist +
                "\n=== FINAL ROUND ===\n"
                "This is your LAST turn. You have asked enough questions.\n"
                "You MUST output your diagnosis NOW. Do NOT ask another question.\n"
                "Output ONLY: DIAGNOSIS READY: [single disease name]\n"
                "Doctor: "
            )
        elif self.prompt_version == "original":
            prompt = (
                "Here is a history of your dialogue: " + self.agent_hist +
                "\nHere was the patient response: " + patient_response +
                "\nNow please continue your dialogue\n"
                "Doctor: "
            )
        else:
            prompt = (
                "=== DIALOGUE HISTORY ===\n" + self.agent_hist +
                "\n=== PATIENT SAID ===\n" + patient_response +
                "\n=== YOUR TASK ===\n"
                "Think step by step:\n"
                "1. What key symptoms and findings do I have so far?\n"
                "2. What information am I still missing?\n"
                "3. What is my next question or test? (Must be NEW, not repeated)\n"
                "4. Am I confident enough for a final diagnosis?\n\n"
                "If yes to step 4, output: DIAGNOSIS READY: [single disease name]\n"
                "Otherwise, ask your next question or request a test.\n"
                "Doctor: "
            )
        answer = query_model(self.backend, prompt, self.system_prompt(),
                            image_requested=image_requested, scene=self.scenario)
        if answer is None:
            answer = "I need more information. Can you tell me more about your symptoms?"
        self.agent_hist += patient_response + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer

    def col_system_prompt(self):
        return (
            "You are a panel of 3 medical specialists discussing a patient case.\n\n"
            "TASK: Review all symptoms, test results, and history. Reach a consensus on the SINGLE most likely diagnosis.\n\n"
            "RULES:\n"
            "1. Each doctor states their top diagnosis with brief evidence.\n"
            "2. Respond as: Doctor X: [diagnosis] - [key evidence]\n"
            "3. After all opinions, output ONE final answer.\n"
            "4. The final answer MUST be a single disease name, not a list.\n"
            "5. End with: DIAGNOSIS READY: [single disease name]\n\n"
            "Do NOT say 'or', 'versus', or 'need more tests'. Pick ONE diagnosis."
        )

    def internal_discussion(self, patient_statement, image_requested = None, scenario_id=0, thread_id=1)->str:
        """
        Simulates an internal multi-doctor discussion to refine diagnosis.

        Args:
            patient_statement (str): The patient's latest statement or response.
            context (str): Additional context (e.g., test results or history).

        Returns:
            str: The final diagnosis after internal discussion.
        """
        discussion_prompt = (
            f"Patient Statement: {patient_statement}\n"
            f"Additional Context: {self.agent_hist}\n"
            f"Doctors, please discuss this case and provide your final diagnosis. "
            f"You MUST include 'DIAGNOSIS READY:' followed by the disease name at the end of your response. "
        )
        responses = []

        for i in range(1, self.num_doctors + 1):
            prompt = (f"{discussion_prompt} Doctor {i}, please share your opinion on the diagnosis and reasoning.\n")
            response = query_model(self.backend, prompt, self.col_system_prompt(), image_requested=image_requested, scene=self.scenario)
            responses.append(f"Doctor {i}: {response.strip()}")

        # Final consensus: synthesize all doctor opinions into one diagnosis
        consensus_prompt = (
            f"The following discussion occurred among {self.num_doctors} doctors:\n\n"
            + "\n".join(responses)
            + "\n\nBased on the discussion above, synthesize a final diagnosis. "
            "You MUST include 'DIAGNOSIS READY:' followed by the single most likely disease name."
        )
        final_response = query_model(self.backend, consensus_prompt, self.col_system_prompt(), image_requested=None, scene=self.scenario)
        self.agent_hist += "\n".join(responses) + f"\nFinal Diagnosis: {final_response.strip()}\n"

        return final_response.strip()

    def inference_final_disease_prediction(self, patient_response: str, image_requested: bool = False) -> str:
        """
        Conducts a multi-round, multi-agent debate for the final disease prediction, incorporating
        more structured prompts, a contradiction-finder, and final consensus summarization.
        """
        # Start with the dialogue history plus the latest patient response.
        debate_history = self.agent_hist + "\nPatient response: " + patient_response + "\n\n"

        # Define the agents (including a Contradiction-Finder).
        agents = ["General Practitioner", "Specialist", "Contradiction-Finder"]
        agent_opinions = {}

        # Round 1: Each agent provides an initial diagnosis with evidence and confidence.
        for agent in agents:
            prompt = (
                debate_history
                + f"{agent}: Based on the conversation so far, please provide:\n"
                + "1) Your top 1-3 possible diagnoses.\n"
                + "2) A brief list of the key clinical findings from the conversation.\n"
                + "3) A confidence score or probability for each diagnosis.\n"
                + "4) Any immediate contradictions or missing data.\n\n"
                + f"{agent}: "
            )
            answer = query_model(self.backend, prompt, self.system_prompt(),
                                image_requested=image_requested, scene=self.scenario)
            agent_opinions[agent] = (answer or "").strip()

        # Debate Rounds: Agents critique each other’s opinions and update.
        rounds = 2
        for r in range(rounds):
            new_opinions = {}
            for agent in agents:
                # Compile others’ opinions
                other_opinions = "\n".join([
                    f"{other} said:\n{agent_opinions[other]}"
                    for other in agents if other != agent
                ])
                prompt = (
                    debate_history
                    + f"Debate Round {r+2}:\n"
                    + "Below are the current opinions from the other agents:\n"
                    + f"{other_opinions}\n\n"
                    + f"{agent}: Please update your diagnosis considering:\n"
                    + "1) The other agents’ points.\n"
                    + "2) Any logical gaps or contradictions.\n"
                    + "3) Any overlooked tests or missing data.\n"
                    + "Provide revised diagnoses, each with a brief rationale.\n\n"
                    + f"{agent}: "
                )
                answer = query_model(self.backend, prompt, self.system_prompt(),
                                    image_requested=image_requested, scene=self.scenario)
                new_opinions[agent] = (answer or "").strip()
            agent_opinions = new_opinions

        # Final Consensus: Summarize all agents’ final opinions
        opinions_summary = "\n\n".join([
            f"{agent} final opinion:\n{opinion}"
            for agent, opinion in agent_opinions.items()
        ])

        # Here we explicitly request that the final output be in "DIAGNOSIS READY: <diagnosis>" format.
        final_prompt = (
            debate_history
            + "=== Final Debate Summary ===\n"
            + opinions_summary
            + "\n\nNow, based on the above expert discussion, provide a single unified consensus diagnosis. "
            "Please format your final answer as:\n\nDIAGNOSIS READY: [your final diagnosis here].\n\n"
            + "DIAGNOSIS READY: "
        )

        final_diagnosis = query_model(self.backend, final_prompt, self.system_prompt(),
                                    image_requested=image_requested, scene=self.scenario)

        # Update the dialogue history with the patient response and the final diagnosis.
        self.agent_hist += (
            patient_response
            + "\n\nFinal Disease Prediction: "
            + final_diagnosis
            + "\n\n"
        )
        self.infs += 1
        return final_diagnosis
    def reset(self) -> None:
        """Resets the doctor agent state."""
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()


class MeasurementAgent:
    """Simulates a measurement reader (for test results) in the dialogue simulation."""

    def __init__(self, scenario=None, backend_str="gpt4", prompt_version="optimized"):
        self.agent_hist = ""
        self.presentation = ""
        self.backend = backend_str
        self.scenario = scenario
        self.prompt_version = prompt_version
        if scenario is not None:
            self.reset()

    def update_scenario(self, scenario):
        self.scenario = scenario
        self.reset()

    def system_prompt(self) -> str:
        """Builds the system prompt for the measurement agent."""
        if self.prompt_version == "original":
            base = (
                "You are an measurement reader who responds with medical test results. "
                "Please respond in the format 'RESULTS: [results here]'.\n\n"
                f"Below is the available exam information:\n{self.information}\n\n"
                "If the requested results are not in your data then respond with 'NORMAL READINGS'."
            )
            return base
        else:
            base = (
                "You are a medical test result reporter. You provide test results to a doctor.\n\n"
                "RULES:\n"
                "1. ONLY report results that exist in the exam data below.\n"
                "2. If the requested test IS in your data, report ALL details for that test.\n"
                "3. If the requested test is NOT in your data, respond with: RESULTS: NORMAL READINGS\n"
                "4. Do NOT interpret results or suggest diagnoses.\n"
                "5. Do NOT answer clinical questions. Only provide test results.\n\n"
                "FORMAT:\n"
                "RESULTS: [exact findings from your data]\n\n"
                "IMPORTANT:\n"
                "- Read your data carefully before responding.\n"
                "- Report the actual values and findings, not summaries.\n"
                "- Include all abnormal and relevant normal findings for the requested test."
            )
            presentation = (
                f"\n\nAvailable exam data:\n{self.information}\n\n"
                "Report results ONLY from this data."
            )
            return base + presentation

    def inference_measurement(self, request: str) -> str:
        """Queries the language model for test result data."""
        if self.prompt_version == "original":
            prompt = (
                "Here is a history of the dialogue: " + self.agent_hist +
                "\nHere was the doctor measurement request: " + request
            )
        else:
            prompt = (
                "=== DIALOGUE HISTORY ===\n" + self.agent_hist +
                "\n=== DOCTOR REQUESTED ===\n" + request +
                "\n=== YOUR TASK ===\n"
                "Check if the requested test exists in your exam data.\n"
                "If YES: report the exact findings.\n"
                "If NO: respond with RESULTS: NORMAL READINGS\n"
                "Do NOT add clinical interpretations.\n"
                "Measurement: "
            )
        answer = query_model(self.backend, prompt, self.system_prompt())
        if answer is None:
            answer = "RESULTS: NORMAL READINGS"
        self.agent_hist += request + "\n\n" + answer + "\n\n"
        return answer

    def add_hist(self, hist_str: str) -> None:
        """Adds additional history."""
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        """Resets the measurement agent state."""
        self.agent_hist = ""
        self.information = self.scenario.exam_information()

def compare_results(diagnosis, correct_diagnosis, mod_pipe, similarity_threshold=0.8, tries=3, timeout=10.0):
    """
    Compares the doctor's diagnosis with the correct diagnosis using a similarity-based approach.

    Args:
        diagnosis (str): The diagnosis provided by the doctor.
        correct_diagnosis (str): The correct diagnosis for the case.
        mod_pipe: Moderator LLM string (e.g. "zhipu:glm-4.7-flash") or BAgent instance.
        similarity_threshold (float): Threshold for similarity to decide "Yes". Defaults to 0.8.
        tries (int): Number of retry attempts. Defaults to 3.
        timeout (float): Time in seconds between retries. Defaults to 5.0.

    Returns:
        bool: True if similarity >= threshold.
    """
    prompt = (
        f"Correct diagnosis: {correct_diagnosis}\n"
        f"Doctor's diagnosis: {diagnosis}\n\n"
        "Compare these two diagnoses. Are they the same disease?\n"
        "Consider that the same disease may have different names, abbreviations, or spellings.\n"
        "Respond with ONLY a similarity score in brackets.\n"
        "Format: [0.XX]\n"
        "Example: [0.95]"
    )
    system_prompt = (
        "You are a medical expert comparing two diagnoses for similarity.\n\n"
        "TASK: Rate how similar the two diagnoses are on a scale from 0.00 to 1.00.\n\n"
        "SCALE:\n"
        "- 1.00 = Same disease (exact match or synonyms, e.g. 'Myasthenia Gravis' vs 'Myasthenia gravis')\n"
        "- 0.80-0.99 = Very close (same condition, different specificity, e.g. 'Acute myeloid leukemia' vs 'AML')\n"
        "- 0.40-0.79 = Partially similar (same organ system or category, e.g. 'pneumonia' vs 'lung cancer')\n"
        "- 0.00-0.39 = Different diseases\n\n"
        "Respond with ONLY a number in brackets. Example: [0.85]\n"
        "Do NOT include any text, explanation, or words."
    )

    for attempt in range(tries):
        try:
            # Use global query_model if mod_pipe is a string, otherwise use BAgent
            if isinstance(mod_pipe, str):
                response = query_model(mod_pipe, prompt, system_prompt)
            else:
                response = mod_pipe.query_model(prompt=prompt, system_prompt=system_prompt, tries=1, timeout=timeout)
            print(f"Attempt {attempt + 1} response: {response}")

            if response and response.startswith("[") and response.endswith("]"):
                similarity_str = response[1:-1]
                similarity = float(similarity_str)
                if 0 <= similarity <= 1:
                    return similarity >= similarity_threshold
                else:
                    print(f"Invalid similarity score: {similarity}")
            else:
                print(f"Response not in expected format: {response}")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")

        time.sleep(timeout)

    logger.warning(f"Failed to compare results after {tries} attempts. Defaulting to INCORRECT.")
    return False
