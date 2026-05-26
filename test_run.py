"""Quick end-to-end test: 1 scenario, 5 rounds."""
import sys
sys.path.insert(0, ".")

from medsim.core.scenario import ScenarioLoaderMedQA
from medsim.core.agent import MeasurementAgent, PatientAgent, DoctorAgent, compare_results
from medsim.query_model import BAgent

loader = ScenarioLoaderMedQA()
scenario = loader.get_scenario(0)
backend = "qwue7"

print("Creating agents...")
meas = MeasurementAgent(backend_str=backend)
patient = PatientAgent(backend_str=backend)
doctor = DoctorAgent(backend_str=backend)
mod = BAgent()

# Update scenario for each agent
meas.update_scenario(scenario=scenario)
patient.update_scenario(scenario=scenario, bias_present=None)
doctor.update_scenario(scenario=scenario, max_infs=5, bias_present=None, img_request=False)

print(f"Scenario diagnosis: {scenario.diagnosis_information()}")
print("Starting simulation...\n")

pi_dialogue = ""
for i in range(5):
    print(f"--- Round {i+1} ---")
    doctor_resp = doctor.inference_doctor(pi_dialogue)
    print(f"Doctor: {doctor_resp}\n")

    if "DIAGNOSIS READY" in doctor_resp:
        correct = compare_results(doctor_resp, scenario.diagnosis_information(), mod)
        print(f"Correct answer: {scenario.diagnosis_information()}")
        print(f"Result: {'CORRECT' if correct else 'INCORRECT'}")
        break

    if "REQUEST TEST" in doctor_resp:
        pi_dialogue = meas.inference_measurement(doctor_resp)
        print(f"Measurement: {pi_dialogue}\n")
        patient.add_hist(pi_dialogue)
    else:
        pi_dialogue = patient.inference_patient(doctor_resp)
        print(f"Patient: {pi_dialogue}\n")
        meas.add_hist(pi_dialogue)

print("\nSimulation complete.")
