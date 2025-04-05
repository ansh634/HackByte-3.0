import io
from matplotlib import scale
import streamlit as st
import requests
import json
import base64
import pytesseract
import time
import cv2
from PIL import Image
import altair as alt
import pandas as pd  # Added import for pandas
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import re


# Sidebar titles
st.sidebar.title("NutriScan Dashboard")
st.sidebar.title("Nutritional Information")
st.sidebar.title("Assessment")
st.sidebar.title("NutriScore")
st.sidebar.title("NutriChat Bot")

# Function to process camera input
def process_camera_input():
    try:
        # Initialize camera capture
        camera_input = st.camera_input("Take a picture of the nutrition label")
        
        if camera_input is not None:
            # Convert the camera input to bytes
            bytes_data = camera_input.getvalue()
            
            # Convert to PIL Image
            image = Image.open(BytesIO(bytes_data))
            
            # Convert PIL image to numpy array
            image_array = np.array(image)
            
            # Basic image preprocessing
            if len(image_array.shape) == 3:  # Color image
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:  # Already grayscale
                gray = image_array
                
            # Enhance contrast
            enhanced = cv2.equalizeHist(gray)
            
            # Apply threshold
            _, threshold = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(threshold)
            
            # Convert to bytes for return
            img_byte_arr = BytesIO()
            processed_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return img_byte_arr, image
            
    except Exception as e:
        st.error(f"Error processing camera input: {str(e)}")
        return None, None
    
    return None, None

# Function to calculate BMI
def calculate_bmi(weight, height_m):
    return weight / (height_m ** 2)

def display_nutrition_chart(nutrition_info):
    # Convert nutrition information into a list for plotting
    nutrients = ["Fat", "Protein", "Carbohydrates", "Sugar", "Fibre", "Sodium", "Potassium", "Calcium"]
    values = [
        nutrition_info["fat"]["total"]["quantity"],
        nutrition_info["protein"]["quantity"],
        nutrition_info["carbohydrates"]["quantity"],
        nutrition_info["Sugar"]["quantity"],
        nutrition_info["fibre"]["quantity"],
        nutrition_info["Sodium"]["quantity"],
        nutrition_info["Potassium"]["quantity"],
        nutrition_info["calcium"]["quantity"]
    ]

    plt.style.use("seaborn-v0_8")  # Use 'seaborn' style; you can change it to any other style you like

    # Convert nutrition data to pandas DataFrame for easy plotting
    nutrition_df = pd.DataFrame(nutrition_info).T
    # Replace non-numeric 'quantity' entries with 0, or use pd.to_numeric with errors='coerce'
    nutrition_df['quantity'] = pd.to_numeric(nutrition_df['quantity'], errors='coerce').fillna(0)

    
    # Plotting the nutritional data with thel new style
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(nutrition_df.index, nutrition_df['quantity'], color="dodgerblue", edgecolor="black")
    ax.set_xlabel("Nutrient", fontsize=12, weight='bold')
    ax.set_ylabel("Quantity (per serving)", fontsize=12, weight='bold')
    ax.set_title("Nutritional Content of Food Product", fontsize=14, weight='bold')
    plt.xticks(rotation=45, ha='right')
    
    st.pyplot(fig)
    
    # Data for Altair chart
    chart_data = pd.DataFrame({
        'Nutrient': nutrients,
        'Value': values
    })

    # Create a bar chart
    chart = alt.Chart(chart_data).mark_bar().encode(
        x='Nutrient',
        y='Value'
    ).properties(
        title='Nutritional Content per 100g'
    )
    
    st.altair_chart(chart, use_container_width=True)

# Function to get personal health impact
def get_health_impact(user_data, nutrition_info):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",
    }
    
    # Convert nutrition_info to string format
    nutrition_str = ", ".join([f"{k}: {v}" for k, v in nutrition_info.items()])
    
    data = json.dumps({
        "model": "google/gemini-2.5-pro-exp-03-25:free",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"You are given age {user_data['age']}, gender {user_data['gender']}, "
                    f"height {user_data['height']:.2f} meters, weight {user_data['weight']} kg, "
                    f"and the nutritional content information of packaged food: {nutrition_str}. "
                    f"Show how this is going to impact the user. Show expected BP and expected sugar level "
                    f"after eating the packaged food. Give the answer in this format: "
                    f"EXPECTED SUGAR LEVEL: VAL,\nEXPECTED BP LEVEL: VAL;"
                )
            }
        ]
    })

    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, data=data)
            if response.status_code == 200:
                response_json = response.json()
                if 'choices' in response_json and response_json['choices']:
                    return response_json['choices'][0]['message']['content']
            time.sleep(1)
        except Exception as e:
            st.error(f"Error in API call: {str(e)}")
    return None

# Function to extract nutrition info from image
def extract_nutrition_from_image(image_data):
    base64_image = base64.b64encode(image_data).decode("utf-8")
    base64_image_data = f"data:image/jpeg;base64,{base64_image}"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",
    }

    payload = json.dumps({
        "model": "google/gemini-2.5-pro-exp-03-25:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                        The provided Image contains nutritional information for a packaged food product. 
                        Extract the information from the image and return it in JSON format. Return only the JSON, 
                        nothing else. The json should be of the following format 
                        
                        {
                            num_servings: value,
                            serving_size: value,
                            fat: {
                            total : {quantity: value, dv: value},
                            saturated: {quantity: value, dv: value},
                            unsaturated: {quantity: value, dv: value},
                            transfat: {quantity: value, dv:value}
                            },
                            protein: {quantity: value, dv: value}, 
                            carbohydrates: {quantity: value, dv: value},
                            Sugar: {quantity: value, dv: value},
                            fibre: {quantity: value, dv: value},
                            Cholestrol: {quantity: value, dv: value}, 
                            Sodium: {quantity: value, dv: value},
                            Potassium: {quantity: value, dv: value},
                            calcium: {quantity: value, dv: value},
                        }
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image_data
                        }
                    }
                ]
            }
        ]
    })

    while True:
        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()
            response_data = response.json()

            if "choices" in response_data and len(response_data["choices"]) > 0:
                break

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying...")
        except json.JSONDecodeError:
            print("Failed to parse JSON response. Retrying...")
        time.sleep(0.5)

    content = response_data['choices'][0]['message']['content']
    cleaned_json_string = content.replace('```json\n', '').replace('```', '').strip()
    return json.loads(cleaned_json_string)

# Function to get health assessment
def get_health_assessment(nutrition_info):
    attempt = 0
    while attempt < 50:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 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",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": "google/gemini-2.5-pro-exp-03-25:free",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"""Pretend that you are a health specialist named: Dr. Prachi.
                            You are provided with the nutritional information of a packaged food item in JSON format. 
                            Your task is to provide a full assessment of the food item, mention its nutritional downsides and upsides(Keep it Pointwise and precise and short): {json.dumps(nutrition_info)}"""
                        }
                    ]
                })
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    print("Error: Unexpected response format. Retrying ...")
            else:
                print(f"Error: Failed to get a valid response. Status code: {response.status_code}. Retrying...")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}. Retrying...")
        except ValueError as e:
            print(f"Failed to parse JSON response: {str(e)}. Retrying...")

        attempt += 1
        time.sleep(0.5)
def extract_nutriscore_grade(nutri_score_text):
    """Extract the grade letter from the NutriScore calculation text"""
    match = re.search(r'NutriScore:\s*([A-E])', nutri_score_text)
    if match:
        return match.group(1)
    return None

def display_nutriscore_visual(grade):
    """Display the visual NutriScore grade circles"""
    colors = {
        'A': '#038141',  # Dark Green
        'B': '#85BB2F',  # Light Green
        'C': '#FECB02',  # Yellow
        'D': '#EE8100',  # Orange
        'E': '#E63E11'   # Red
    }
    
    cols = st.columns(5)
    
    for i, (letter, color) in enumerate(colors.items()):
        with cols[i]:
            if letter == grade:
                # Highlighted style for the actual grade
                st.markdown(
                    f"""
                    <div style="
                        width: 60px;
                        height: 60px;
                        border-radius: 50%;
                        background-color: {color};
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-weight: bold;
                        font-size: 24px;
                        margin: auto;
                        box-shadow: 0 0 10px rgba(0,0,0,0.3);
                        border: 3px solid white;
                        animation: pulse 2s infinite;
                    ">
                        {letter}
                    </div>
                    <style>
                       
 @keyframes pulse {{
                        0% {{ transform:scale(1); }}
                        50% {{ transform:scale(1.1); }}
                        100% {{ transform:scale(1); }}
                    }}


                    </style>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                # Dimmed style for other grades
                st.markdown(
                    f"""
                    <div style="
                        width: 50px;
                        height: 50px;
                        border-radius: 50%;
                        background-color: {color};
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-weight: bold;
                        font-size: 20px;
                        margin: auto;
                        opacity: 0.5;
                    ">
                        {letter}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

# Main application
st.title("NutriScan: Your Pocket Nutritionist")
tabs = st.tabs(["Nutrition Scanner", "NutriScore Calculator", "Personal Health Profile"])

# First tab: Nutrition Scanner
with tabs[0]:
    st.markdown('<p style="font-size:18px;">Take a picture or upload an image of a food product\'s nutritional label to get a detailed nutritional breakdown and health assessment!</p>', unsafe_allow_html=True)
    
    # Image capture section
    st.subheader("Capture or Upload Image")
    image_source = st.radio("Choose image source:", ["Camera", "Upload"])
    
    image_data = None
    display_image = None
    
    if image_source == "Camera":
        processed_image_data, original_image = process_camera_input()
        if processed_image_data is not None:
            image_data = processed_image_data
            display_image = original_image
            st.success("Image captured successfully!")
    else:
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_data = uploaded_file.getvalue()
            display_image = Image.open(uploaded_file)
            st.success("Image uploaded successfully!")
    
    if display_image is not None:
        st.image(display_image, caption="Captured/Uploaded Image", use_container_width=True)
        
        with st.spinner('Extracting nutritional information...'):
            nutrition_info = extract_nutrition_from_image(image_data)
            st.success('Nutritional information extracted successfully!')
            st.json(nutrition_info)
            
            st.write("Visualizing Nutritional Content:")
            display_nutrition_chart(nutrition_info)
            
            st.write("Health Assessment by Dr. Prachi:")
            with st.spinner('Analyzing the food item...'):
                health_assessment = get_health_assessment(nutrition_info)
                st.write(health_assessment)

# Second tab: NutriScore Calculator
with tabs[1]:
    st.header("NutriScore Calculator")
    st.markdown('<p style="font-size:18px;">Calculate the NutriScore for your food product!</p>', unsafe_allow_html=True)
    
    # Option to use previously scanned nutrition info or enter new values
    score_source = st.radio(
        "Choose data source for NutriScore calculation:",
        ["Use scanned nutrition information", "Enter values manually"]
    )
    
    if score_source == "Use scanned nutrition information":
        if 'nutrition_info' in locals():
            if st.button("Calculate NutriScore from Scanned Data"):
                with st.spinner("Calculating NutriScore..."):
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": "Bearer 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",
                            "Content-Type": "application/json"
                        },
                        data=json.dumps({
                            "model": "google/gemini-2.5-pro-exp-03-25:free",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"""
                                    You are a NutriScore Calculator. 
                                    Here is the nutritional information of a packaged food item in JSON format:
                                    {json.dumps(nutrition_info)}
                                    calculate the nutriscore based on the provided data. return the result in format with the following structure.
                                     
                                    Negative points:
                                    - Energy: Value
                                    - Sugars: Value 
                                    - Saturated fatty acids: Value 
                                    - Sodium: Value 
                                    Total negative points: Value 

                                    Positive points:
                                    - Fiber: Value 
                                    - Protein: Value 
                                    Total positive points: Value 

                                    NutriScore (letter grade only).
                                    """
                                }
                            ]
                        })
                    )

                    if response.status_code == 200:
                        response_content = response.json()
                        if 'choices' in response_content:
                            nutri_score = response_content['choices'][0]['message']['content']
                            st.write(nutri_score)
                            st.success(f"The NutriScore calculation is complete!")
                    else:
                        st.error("Error: Unable to calculate NutriScore. Please try again.")
        else:
            st.warning("Please scan a nutrition label first in the Nutrition Scanner tab.")
    
    else:
        # Manual input fields for NutriScore calculation
        st.subheader("Enter Nutritional Values")
        col1, col2 = st.columns(2)
        
        with col1:
            energy = st.number_input("Energy (kcal/100g)", min_value=0.0, step=0.1)
            sugars = st.number_input("Sugars (g/100g)", min_value=0.0, step=0.1)
            saturated_fat = st.number_input("Saturated Fat (g/100g)", min_value=0.0, step=0.1)
        
        with col2:
            sodium = st.number_input("Sodium (mg/100g)", min_value=0.0, step=0.1)
            fiber = st.number_input("Fiber (g/100g)", min_value=0.0, step=0.1)
            protein = st.number_input("Protein (g/100g)", min_value=0.0, step=0.1)
        
if st.button("Calculate NutriScore from Manual Input"):
        manual_nutrition_info = {
            "energy": {"quantity": energy},
            "Sugar": {"quantity": sugars},
            "fat": {"saturated": {"quantity": saturated_fat}},
            "Sodium": {"quantity": sodium},
            "fibre": {"quantity": fiber},
            "protein": {"quantity": protein}
        }
        
        with st.spinner("Calculating NutriScore..."):
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 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",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": "google/gemini-2.5-pro-exp-03-25:free",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"""
                            You are a NutriScore Calculator. 
                            Here is the nutritional information of a packaged food item in JSON format:
                            {json.dumps(manual_nutrition_info)}
                            calculate the nutriscore based on the provided data. return the result in format with the following structure.
                             
                            Negative points:
                            - Energy: Value
                            - Sugars: Value 
                            - Saturated fatty acids: Value 
                            - Sodium: Value 
                            Total negative points: Value 

                            Positive points:
                            - Fiber: Value 
                            - Protein: Value 
                            Total positive points: Value 

                            NutriScore: [A/B/C/D/E]
                            """
                        }
                    ]
                })
            )

            if response.status_code == 200:
                response_content = response.json()
                if 'choices' in response_content:
                    nutri_score_text = response_content['choices'][0]['message']['content']
                    
                    # Display the detailed calculation
                    st.write("### Calculation Details")
                    st.text(nutri_score_text)
                    
                    # Extract and display the grade visually
                    grade = extract_nutriscore_grade(nutri_score_text)
                    if grade:
                        st.write("### NutriScore Grade")
                        display_nutriscore_visual(grade)
                        
                        # Add explanatory text based on grade
                        explanations = {
                            'A': "Excellent nutritional quality! This product is among the healthiest choices.",
                            'B': "Good nutritional quality. This product contributes to a balanced diet.",
                            'C': "Moderate nutritional quality. Consider portion sizes and frequency of consumption.",
                            'D': "Lower nutritional quality. Consider healthier alternatives when possible.",
                            'E': "Lowest nutritional quality. Consume in moderation and less frequently."
                        }
                        
                        st.info(explanations.get(grade, ""))
                        
                    st.success(f"The NutriScore calculation is complete!")
            else:
                st.error("Error: Unable to calculate NutriScore. Please try again.")
# Third tab: Personal Health Profile
with tabs[2]:
    st.header("Personal Health Profile")
    st.markdown("Enter your personal information to get customized health impact analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=300.0, value=170.0, step=0.1)
    
    with col2:
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=500.0, value=70.0, step=0.1)
        activity_level = st.select_slider(
            "Activity Level",
            options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"]
        )
    
    # Calculate BMI
    height_m = height_cm / 100
    bmi = calculate_bmi(weight, height_m)
    
    # Display BMI in a metric
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("BMI", f"{bmi:.1f}")
    with col2:
        bmi_category = (
            "Underweight" if bmi < 18.5
            else "Normal weight" if bmi < 25
            else "Overweight" if bmi < 30
            else "Obese"
        )
        st.metric("BMI Category", bmi_category)
    
    # Personal Health Impact Analysis
    if st.button("Analyze Health Impact", type="primary"):
        user_data = {
            "age": age,
            "gender": gender,
            "height": height_m,
            "weight": weight,
            "activity_level": activity_level
        }
        
        with st.spinner("Analyzing personal health impact..."):
            if 'nutrition_info' in locals():
                impact_result = get_health_impact(user_data, nutrition_info)
                if impact_result:
                    st.success("Analysis Complete!")
                    
                    with st.expander("View Detailed Health Impact", expanded=True):
                        st.write(impact_result)
                        
                        try:
                            sugar_level = impact_result.split("EXPECTED SUGAR LEVEL:")[1].split(",")[0].strip()
                            bp_level = impact_result.split("EXPECTED BP LEVEL:")[1].split(";")[0].strip()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Expected Sugar Level", sugar_level)
                            with col2:
                                st.metric("Expected BP Level", bp_level)
                        except:
                            st.warning("Could not parse detailed metrics")
            else:
                st.warning("Please scan or upload a nutrition label first to get personalized health impact analysis")
    pass
# NutriChat Bot
st.header("NutriChat Bot")
st.markdown('<p style="font-size:18px;">Ask NutriChat Bot any questions about nutrition or fitness!</p>', unsafe_allow_html=True)

# Initialize the conversation history
msgs = [
    {
        "role": "user",
        "content": "Pretend that you are a trained nutritionist who is very good at helping people achieve their fitness goals. Respond as the nutritionist and answer all questions. The replies should be brief and conversational."
    }
]

# User input for NutriChat Bot
user_prompt = st.text_input("Enter your question:")

if user_prompt:
    msgs.append({"role": "user", "content": user_prompt})
    
    response_data = None
    while not response_data:
        try:
            response_data = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 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",
                },
                data=json.dumps({
                    "model": "google/gemini-2.5-pro-exp-03-25:free",
                    "messages": msgs
                })
            ).json()

            if 'choices' in response_data and response_data['choices']:
                chat_response = response_data['choices'][0]['message']['content']
                st.write(chat_response)
                msgs.append({"role": "assistant", "content": chat_response})
            else:
                response_data = None

        except requests.exceptions.RequestException as e:
            st.write(f"An error occurred: {e}")
            response_data = None

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    # Encode your local image
    img_file = get_base64("/Users/91876/nutriscanner/bg.jpg")
    
    # Inject CSS with the base64 string
    page_bg_img = f'''
    <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_file}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
except:
    st.warning("Background image not found. Using default background.")