import os
import google.generativeai as genai
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="wide")

os.environ["GEMINI_API_KEY"] = "AIzaSyAGqESvPcj1iSCUy3Wi5hnBPKCtgsPYK7Q"  # Replace with your API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

@st.cache_data
def load_models_and_data():
    kmeans = joblib.load(r"C:\Users\navee\OneDrive\Desktop\Clustering\required_pkl\kmeans_model.pkl")
    cuisines_encoder = joblib.load(r"C:\Users\navee\OneDrive\Desktop\Clustering\required_pkl\encoded_cuisines.pkl")
    city_encoder = joblib.load(r"C:\Users\navee\OneDrive\Desktop\Clustering\required_pkl\label_encoder_city.pkl")
    scaler = joblib.load(r"C:\Users\navee\OneDrive\Desktop\Clustering\required_pkl\scaler.pkl")
    Final_df = pd.read_csv(r"C:\Users\navee\OneDrive\Desktop\Clustering\Processed_Data\Final_df1.csv")
    return kmeans, cuisines_encoder, city_encoder, scaler, Final_df

kmeans, cuisines_encoder, city_encoder, scaler, Final_df = load_models_and_data()

required_columns = ['City', 'Cuisines', 'Rating', 'Rating_color', 'Average_cost_for_two_usd', 'Featured_image', 'Name', 'Currency', 'Address', 'Url']
missing_columns = [col for col in required_columns if col not in Final_df.columns]
if missing_columns:
    st.error(f"Missing required columns in dataset: {', '.join(missing_columns)}")
    st.stop()

def chat_with_gemini(prompt):
    """
    Query the Google Gemini model with a food-specific system prompt.
    """
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
            system_instruction="You are a helpful assistant specialized in food, cuisines, and recipes. Answer only in this context."
        )
        chat_session = model.start_chat()
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

st.title("Restaurant Recommendation System")
st.write("Find the best restaurants based on your preferences!")

st.write("### Ask the Assistant")
user_query = st.text_input("Type your question about food, cuisines, or recipes:")

if user_query:
    st.write("**Assistant Response:**")
    with st.spinner("Thinking..."):
        llm_response = chat_with_gemini(user_query)
    st.write(llm_response)

st.write("### Search Filters")
col1, col2, col3 = st.columns(3)

with col1:
    city_input = st.selectbox("Select a city:", Final_df['City'].unique())

available_cuisines = []
if city_input:
    available_cuisines = list(
        set(", ".join(Final_df[Final_df['City'] == city_input]['Cuisines']).split(", ")
        )
    )

cuisine_query = st.selectbox("Search for cuisines:", [""] + available_cuisines, help="Select or type a cuisine to get suggestions.")

budget_options = {
    "Budget-Friendly": (0, 20),
    "Moderate-Price": (20, 50),
    "Premium": (50, Final_df['Average_cost_for_two_usd'].max())
}
with col2:
    budget_choice = st.selectbox("Select price range:", options=list(budget_options.keys()))
min_budget, max_budget = budget_options[budget_choice]

with col3:
    rating_input = st.slider(
        "Select minimum rating:",
        float(Final_df['Rating'].min()),
        float(Final_df['Rating'].max()),
        4.0,
        key="rating_slider"
    )
    slider_color = Final_df[Final_df['Rating'] >= rating_input]['Rating_color'].iloc[0] if not Final_df[Final_df['Rating'] >= rating_input].empty else "000000"
    st.markdown(f"<style>div[data-testid='stSlider'] > div > div > div[role='slider'] {{ background-color: #{slider_color}; }}</style>", unsafe_allow_html=True)

filtered_data = Final_df[
    (Final_df['City'] == city_input) &
    (Final_df['Rating'] >= rating_input) &
    (Final_df['Average_cost_for_two_usd'] >= min_budget) &
    (Final_df['Average_cost_for_two_usd'] <= max_budget)
]

if cuisine_query:
    filtered_data = filtered_data[
        filtered_data['Cuisines'].apply(lambda x: cuisine_query.lower() in [c.lower() for c in x.split(", ")])
    ]

filtered_data = filtered_data.drop_duplicates(subset=['Name', 'Address'])

if 'Rating' in filtered_data.columns:
    filtered_data = filtered_data.sort_values(by='Rating', ascending=False)
else:
    st.error("Rating column is missing from the filtered data.")
    st.stop()

st.write(f"### Recommended Restaurants in {city_input}")

if not filtered_data.empty:
    for index, row in filtered_data.iterrows():
        photo_html = (
            f"<img src='{row['Featured_image']}' alt='Image' width='250' height='200' style='border-radius:8px; margin-right:15px;' />"
            if pd.notna(row['Featured_image']) and row['Featured_image'].strip().startswith("http")
            else "<div style='width:250px; height:200px; background-color:lightgray; border-radius:8px; margin-right:15px; display:flex; justify-content:center; align-items:center;'>No Image</div>"
        )

        restaurant_info = f"""
        <div style="display:flex; align-items:center; padding:15px; margin-bottom:20px; border:1px solid #ccc; border-radius:8px; box-shadow:0 2px 5px rgba(0,0,0,0.1); width:100%;">
            <div style="flex-shrink:0;">
                {photo_html}
            </div>
            <div style="flex-grow:1;">
                <p><strong>Name:</strong> {row['Name']}</p>
                <p><strong>Cuisines:</strong> {row['Cuisines']}</p>
                <p><strong>Cost for Two:</strong> {row['Currency']} {row['Average_cost_for_two_usd']}</p>
                <p><strong>Rating:</strong> <span style='color:#{row['Rating_color'].strip()};'>&#9733; {row['Rating']}</span></p>
                <p><strong>Address:</strong> {row['Address']}</p>
                <a href="{row['Url']}" target="_blank" style="color:#007bff; text-decoration:none;">Visit Restaurant Online</a>
            </div>
        </div>
        """
        st.markdown(restaurant_info, unsafe_allow_html=True)
else:
    st.write("No restaurants match your criteria. Please adjust your filters.")
