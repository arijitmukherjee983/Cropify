# import streamlit as st
# import pandas as pd
# import joblib

# # Load the saved model
# model = joblib.load('random_forest_model.joblib')

# st.title("Crop Recommendation System")

# with st.form(key='recommendation_form'):
#     N = st.number_input('Nitrogen (N)', min_value=0, max_value=140, value=90)
#     P = st.number_input('Phosphorus (P)', min_value=5, max_value=145, value=40)
#     K = st.number_input('Potassium (K)', min_value=5, max_value=205, value=40)
#     temp = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
#     humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=80.0)
#     ph = st.number_input('Soil pH', min_value=0.0, max_value=14.0, value=6.5)
#     rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=300.0, value=200.0)
#     submit_recommend = st.form_submit_button('Recommend Crop')

# if submit_recommend:
#     try:
#         input_df = pd.DataFrame([{
#             'N': N,
#             'P': P,
#             'K': K,
#             'temperature': temp,
#             'humidity': humidity,
#             'ph': ph,
#             'rainfall': rainfall
#         }])

#         prediction = model.predict(input_df)[0]
#         st.write(f"Model predicted label: {prediction}")

#         st.success(f"{prediction} is the best crop to be cultivated right there.")

#     except Exception as e:
#         st.error(f"An error occurred: {e}")

# st.sidebar.title("About")
# st.sidebar.info("""
# This is a crop recommendation app built with Streamlit.
# Provide your soil and weather parameters to get crop suggestions.
# """)

import streamlit as st
import pandas as pd
import joblib
from groq import Groq

def generate_explanation_groq(api_key, inputs, predicted_crop):
    client = Groq(api_key=api_key)
    prompt = (
        f"Given these soil and weather conditions:\n"
        f"Nitrogen={inputs['N']}, Phosphorus={inputs['P']}, Potassium={inputs['K']}, "
        f"Temperature={inputs['temperature']}°C, Humidity={inputs['humidity']}%, "
        f"Soil pH={inputs['ph']}, Rainfall={inputs['rainfall']}mm,\n"
        f"explain why {predicted_crop} would be a suitable crop to cultivate."
    )
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a helpful agricultural assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
        stop=None,
    )
    explanation = completion.choices[0].message.content.strip()
    return explanation

# Load model once
model = joblib.load('random_forest_model.joblib')

st.title("Crop Recommendation System")

with st.form(key='recommendation_form'):
    N = st.number_input('Nitrogen (N)', min_value=0, max_value=140, value=90)
    P = st.number_input('Phosphorus (P)', min_value=5, max_value=145, value=40)
    K = st.number_input('Potassium (K)', min_value=5, max_value=205, value=40)
    temp = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=80.0)
    ph = st.number_input('Soil pH', min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=300.0, value=200.0)
    submit_recommend = st.form_submit_button('Recommend Crop')

if submit_recommend:
    input_dict = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temp,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    st.session_state['input_dict'] = input_dict
    prediction = model.predict(pd.DataFrame([input_dict]))[0]
    st.session_state['prediction'] = prediction

if 'prediction' in st.session_state and 'input_dict' in st.session_state:
    st.write(f"Model predicted label: {st.session_state['prediction']}")
    st.success(f"{st.session_state['prediction']} is the best crop to be cultivated right there.")

    want_explanation = st.checkbox("Want to know about the prediction? mark the check box and Provide your Groq API key below:")

    if want_explanation:
        api_key = st.text_input("Enter your Groq API Key", type="password")
        generate_btn = st.button("Generate Explanation")
        if generate_btn:
            if api_key:
                with st.spinner("Generating explanation..."):
                    try:
                        explanation = generate_explanation_groq(api_key, st.session_state['input_dict'], st.session_state['prediction'])
                        if explanation:
                            st.info(f"Explanation:\n{explanation}")
                        else:
                            st.warning("Please provide a valid Groq API key.")
                    except Exception:
                        st.warning("Please provide a valid Groq API key.")
            else:
                st.warning("Please enter your Groq API key to get the explanation.")



st.sidebar.title("About")
st.sidebar.info("""
This is a crop recommendation app built with Streamlit.
Provide your soil and weather parameters to get crop suggestions.
""")
