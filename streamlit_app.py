import streamlit as st
import openai
import asyncio
import aiohttp
import speech_recognition as sr
from gtts import gTTS, lang
import os
import tempfile
import base64
from deep_translator import GoogleTranslator
import pycountry

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="Clinical Analysis", page_icon="🩺")

# Initialize session state for language selection and other states
if 'language' not in st.session_state:
    st.session_state['language'] = 'en'

if 'clinical_responses' not in st.session_state:
    st.session_state['clinical_responses'] = None

if 'patient_history' not in st.session_state:
    st.session_state['patient_history'] = None

if 'is_recording' not in st.session_state:
    st.session_state['is_recording'] = False

# Function to generate response from OpenAI
async def generate_response(session, prompt):
    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are an AI medical consultant for medical professionals with extensive knowledge and expertise in clinical medicine. Your role is to provide detailed and accurate analysis, diagnosis, and management recommendations for complex patient cases."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0.7
            },
            ssl=False
        ) as response:
            result = await response.json()
            if response.status == 200 and 'choices' in result:
                return result['choices'][0]['message']['content'].strip()
            else:
                st.error(f"Error: {result.get('error', {}).get('message', 'Unknown error')}")
                return "An error occurred or the response format is unexpected."
    except Exception as e:
        st.error(f"Exception: {e}")
        return "An error occurred while generating the response."

# Function to translate text
def translate_text(text, target_language):
    try:
        translation = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translation
    except Exception as e:
        return f"Translation Error: {e}"

# Function to recognize speech using the microphone
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Wait for 5 seconds after you are done speaking")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success("Recognized text: " + text)
            return text
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
    return ""

# Function to convert text to speech
def text_to_speech(text, language='en'):
    if language not in lang.tts_langs():
        st.error(f"Language not supported: {language}")
        return None
    tts = gTTS(text=text, lang=language)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# Function to autoplay audio in Streamlit
def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

# Streamlit UI
st.title("One TheraCure")
st.subheader("Clinical Decision Support")

# Language selection dropdown
st.sidebar.header("Language Settings")
language = st.sidebar.selectbox("Select Language", ["English", "Hindi", "Telugu", "Tamil", "Kannada", "Marathi"])
language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Marathi": "mr"
}
st.session_state.language = language_codes[language]

# Sidebar navigation
st.sidebar.header("Navigation")
nav_options = ["Instructions & Patient History Input", "Clinical Analysis", "Editable EHR"]
selected_option = st.sidebar.radio("Select a section:", nav_options)

if selected_option == "Instructions & Patient History Input":
    st.header("Instructions")
    instructions = """
    Please enter the patient history in the following format:
    1. **Age**
    2. **Sex**
    3. **Relevant Past Medical History**
    4. **Medications**
    5. **Presenting Symptoms**
    6. **Associated Symptoms**
    7. **Descriptions of Relevant Studies (including labs and imaging)**
    8. **The Illness Course**
    9. **Any Additional Information You Might Include When Consulting Another Physician About Your Patient**
    """
    translated_instructions = translate_text(instructions, st.session_state.language)
    st.markdown(translated_instructions.replace("\n", "<br>"), unsafe_allow_html=True)

    st.header("Patient History Input")
    input_method = st.radio("Select input method:", ("Text", "Voice"))

    if input_method == "Text":
        patient_history = st.text_area(
            "Enter/Paste your content here. Press Enter twice to save it.",
            height=400,
            placeholder=translate_text("Enter patient history here...", st.session_state.language)
        )
    else:
        if not st.session_state['is_recording']:
            if st.button("Start Recording"):
                st.session_state['is_recording'] = True
                patient_history = recognize_speech()
                st.session_state['patient_history'] = patient_history
        else:
            if st.button("Stop Recording"):
                st.session_state['is_recording'] = False
        patient_history = st.session_state.get('patient_history', '')

    if st.button(translate_text("Generate Clinical Analysis and EHR", st.session_state.language)):
        if patient_history:
            st.session_state['patient_history'] = patient_history

            # Define prompts for each section
            prompts = {
                "clinical_problem_representation": f"""
                Clinical Problem Representation:
                {patient_history}
                Provide a concise summary of the key clinical issues, findings, and diagnostic challenges presented in the case. Ensure the response integrates the patient's history, symptoms, physical examination, and test results.
                """,
                "most_likely_diagnosis": f"""
                Most Likely Diagnosis:
                {patient_history}
                Provide your primary most likely diagnostic hypothesis, supported by a detailed explanation that integrates the patient's history, symptoms, physical examination, and test results. Include any relevant pathophysiological mechanisms, epidemiological factors, or diagnostic criteria that support your diagnosis.
                """,
                "expanded_differential_diagnosis": f"""
                Expanded Differential Diagnosis:
                {patient_history}
                Provide a comprehensive list of differential diagnoses that should be considered, along with brief explanations for their inclusion in the differential. Evaluate the relative likelihood of each differential diagnosis based on the available information.
                """,
                "alternative_diagnosis": f"""
                Alternative Diagnosis:
                {patient_history}
                Provide a comprehensive list of alternative diagnoses that should be considered, along with brief explanations for their inclusion in alternative. Evaluate the relative likelihood of each alternative diagnosis based on the available information.
                """,
                "clinical_assessment": f"""
                Clinical Assessment:
                {patient_history}
                List specific diagnostic tests, procedures, or imaging studies that you recommend to confirm or rule out the suspected diagnosis, as well as any additional tests needed to evaluate the differential diagnoses or address any remaining diagnostic uncertainties.
                """,
                "clinical_treatment_plan": f"""
                Clinical Treatment Plan:
                {patient_history}
                You are a highly experienced clinician. Based on the detailed patient case provided below, create an elaborate clinical treatment plan that addresses the patient's condition comprehensively. Your response should include the most likely diagnosis, a thorough analysis of the patient's symptoms, a step-by-step treatment plan, and any necessary follow-up actions. Ensure the response is detailed and at least 400 words.
                """,
                "monitoring_and_follow_up": f"""
                Monitoring and Follow-Up:
                {patient_history}
                Provide comprehensive guidelines for monitoring the patient's response to treatment, potential complications or adverse effects to watch for, and recommended follow-up plans, including any necessary specialist referrals or additional testing.
                """
            }

            async def main():
                async with aiohttp.ClientSession() as session:
                    tasks = [generate_response(session, prompt) for prompt in prompts.values()]
                    clinical_responses = await asyncio.gather(*tasks)
                    return clinical_responses

            clinical_responses = asyncio.run(main())

            st.session_state['clinical_responses'] = clinical_responses

elif selected_option == "Clinical Analysis" and st.session_state['clinical_responses']:
    st.sidebar.header("Clinical Analysis")
    analysis_tabs = [
        "Clinical Problem Representation",
        "Most Likely Diagnosis",
        "Expanded Differential Diagnosis",
        "Alternative Diagnosis",
        "Clinical Assessment",
        "Clinical Treatment Plan",
        "Monitoring and Follow-Up"
    ]
    selected_tab = st.sidebar.radio("Navigate:", analysis_tabs)

    st.header(selected_tab)
    tab_index = analysis_tabs.index(selected_tab)
    response_text = st.session_state['clinical_responses'][tab_index]
    translated_response = translate_text(response_text, st.session_state.language)
    st.write(translated_response)

    # Text-to-Speech Button
    if st.button(translate_text("Listen to the response", st.session_state.language)):
        tts_path = text_to_speech(translated_response, language=st.session_state.language)
        if tts_path:
            autoplay_audio(tts_path)

elif selected_option == "Editable EHR" and st.session_state['clinical_responses']:
    st.header("Editable Electronic Health Record")
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        patient_id = st.text_input("Patient ID", "")
        first_name = st.text_input("First Name", "")
        last_name = st.text_input("Last Name", "")
        dob = st.date_input("Date of Birth")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    with col2:
        contact_info = st.text_area("Contact Information", "")
        address = st.text_area("Address", "")
        emergency_contact = st.text_input("Emergency Contact", "")

    st.subheader("Medical History")
    col3, col4 = st.columns(2)

    with col3:
        allergies = st.text_area("Allergies", "")
        past_conditions = st.text_area("Past Medical Conditions", "")
        surgical_history = st.text_area("Surgical History", "")

    with col4:
        medications = st.text_area("Medications", "")
        family_history = st.text_area("Family History", "")

    st.subheader("Current Visit Details")
    col5, col6 = st.columns(2)

    with col5:
        visit_date = st.date_input("Visit Date")
        reason_for_visit = st.text_area("Reason for Visit", st.session_state['patient_history'])
        chief_complaint = st.text_area("Chief Complaint", st.session_state['patient_history'])

    with col6:
        hpi = st.text_area("History of Present Illness (HPI)", st.session_state['patient_history'])
        ros = st.text_area("Review of Systems (ROS)", st.session_state['patient_history'])
        vital_signs = st.text_area("Vital Signs", "")

    st.subheader("Physical Examination")
    col7, col8 = st.columns(2)

    with col7:
        general_appearance = st.text_area("General Appearance", "")
        heent = st.text_area("HEENT (Head, Eyes, Ears, Nose, Throat)", "")
        cardiovascular = st.text_area("Cardiovascular", "")

    with col8:
        respiratory = st.text_area("Respiratory", "")
        abdomen = st.text_area("Abdomen", "")
        musculoskeletal = st.text_area("Musculoskeletal", "")
        neurological = st.text_area("Neurological", "")
        skin = st.text_area("Skin", "")

    st.subheader("Laboratory and Diagnostic Test Results")
    col9, col10 = st.columns(2)

    with col9:
        lab_results = st.text_area("Lab Results", "")
        test_name = st.text_input("Test Name", "")
        result = st.text_input("Result", "")

    with col10:
        normal_range = st.text_input("Normal Range", "")
        imaging_results = st.text_area("Imaging Results", "")
        imaging_type = st.text_input("Imaging Type", "")
        findings = st.text_input("Findings", "")

    st.subheader("Clinical Decision Support")
    col11, col12 = st.columns(2)

    with col11:
        most_likely_diagnosis = st.text_area("Most Likely Diagnosis", st.session_state['clinical_responses'][1])
        expanded_differential_diagnoses = st.text_area("Expanded Differential Diagnoses", st.session_state['clinical_responses'][2])

    with col12:
        assessment_plan = st.text_area("Assessment and Plan", st.session_state['clinical_responses'][4])
        suggested_history_questions = st.text_area("Suggested History Questions", "")
        suggested_physical_exam_maneuvers = st.text_area("Suggested Physical Exam Maneuvers", "")

    st.subheader("Treatment Plan")
    col13, col14 = st.columns(2)

    with col13:
        medications_prescribed = st.text_area("Medications Prescribed", "")
        procedures_performed = st.text_area("Procedures Performed", "")

    with col14:
        follow_up_instructions = st.text_area("Follow-Up Instructions", st.session_state['clinical_responses'][6])
        lifestyle_recommendations = st.text_area("Lifestyle and Dietary Recommendations", "")

    st.subheader("Patient Instructions and Documentation")
    col15, col16 = st.columns(2)

    with col15:
        care_instructions = st.text_area("Care Instructions", "")
        document_uploads = st.text_area("Document Uploads", "")

    with col16:
        save_print_options = st.text_area("Save and Print Options", "")
        documentation_drafts = st.text_area("Documentation Drafts", "")

    st.subheader("Notifications")
    col17, col18 = st.columns(2)

    with col17:
        appointment_reminders = st.text_area("Appointment Reminders", "")
        new_messages = st.text_area("New Messages", "")

    with col18:
        alert_types = st.text_area("Alert Types", "")

    st.subheader("Billing Information")
    col19, col20 = st.columns(2)

    with col19:
        billing_details = st.text_area("Billing Details", "")
        insurance_provider = st.text_input("Insurance Provider", "")
        policy_number = st.text_input("Policy Number", "")

    with col20:
        billing_codes = st.text_area("Billing Codes", "")
        charges = st.text_area("Charges", "")
        payment_status = st.text_area("Payment Status", "")

    st.subheader("Consent Forms")
    col21, col22 = st.columns(2)

    with col21:
        form_description = st.text_area("Form Description", "")
        consent_date = st.date_input("Date")

    with col22:
        patient_signature = st.text_area("Patient Signature", "")
        physician_signature = st.text_area("Physician Signature", "")

else:
    st.error("Please enter the patient history.")
