import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the models and scalers
car_model = pickle.load(open('car_eval_rf.pkl', 'rb'))
car_encoder = pickle.load(open('car_eval_encoder.pkl', 'rb'))
liver_model = pickle.load(open('liver_model_ridge.pkl', 'rb'))
liver_scaler = pickle.load(open('liver_scaler.pkl', 'rb'))

# Sidebar for model selection
model_choice = st.sidebar.radio("Choose a model", ("Car Evaluation", "Alcoholic Beverages Prediction"))

# Function to preprocess input for the liver model
def preprocess_liver_input(input_data):
    df = pd.DataFrame(input_data, index=[0])
    scaled_features = liver_scaler.transform(df)
    return scaled_features

if model_choice == "Car Evaluation":
	# Define the mapping from user-friendly labels to original labels
	def get_original_value(readable):
		readable_to_original = {
			'Very High': 'vhigh',
			'High': 'high',
			'Medium': 'med',
			'Low': 'low',
			'5 or more': '5more',
			'More': 'more',
			'2': '2',
			'3': '3',
			'4': '4',
			'Small': 'small',
			'Medium': 'med',
			'Big': 'big'
		}
		return readable_to_original.get(readable, readable)

	# Define the mapping from model's output labels to readable labels and emojis
	output_label_mapping = {
		'unacc': ('Unacceptable', '👎'),
		'acc': ('Acceptable', '👍'),
		'good': ('Good', '😃'),
		'vgood': ('Very Good', '🌟')
	}

	# Define the mapping from model's output labels to image filenames
	image_mapping = {
		'unacc': 'unacceptable.jpg',
		'acc': 'acceptable.jpg',
		'good': 'good.jpg',
		'vgood': 'verygood.jpg'
	}

	# Define the web app
	st.markdown("# 🚗 Car Evaluation Predictor")

	# Create inputs for each feature the model requires
	col1, col2, col3 = st.columns(3)

	with col1:
		buying = st.selectbox('Buying Price', options=['Very High', 'High', 'Medium', 'Low'])
	with col2:
		maint = st.selectbox('Maintenance Price', options=['Very High', 'High', 'Medium', 'Low'])
	with col3:
		doors = st.selectbox('Number of Doors', options=['2', '3', '4', '5 or more'])

	col1, col2, col3 = st.columns(3)

	with col1:
		persons = st.selectbox('Capacity in Terms of Persons to Carry', options=['2', '4', 'More'])
	with col2:
		lug_boot = st.selectbox('Size of Luggage Boot', options=['Small', 'Medium', 'Big'])
	with col3:
		safety = st.selectbox('Estimated Safety of the Car', options=['Low', 'Medium', 'High'])

	# When the 'Predict' button is clicked, make a prediction and display it
	if st.button('Predict'):
		# Map the readable input features back to their original values
		input_features = [buying, maint, doors, persons, lug_boot, safety]
		input_features_original = [get_original_value(feature) for feature in input_features]

		# Create a DataFrame with the original input features
		input_df_original = pd.DataFrame([input_features_original], columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

		# One-hot encode the input data using the original values
		input_encoded = car_encoder.transform(input_df_original)

		# Make a prediction with the encoded data
		prediction = car_model.predict(input_encoded)

		# Convert the model's prediction to a readable label and emoji
		readable_prediction, emoji = output_label_mapping.get(prediction[0], ("Unknown", "❓"))

		# Display the readable prediction with an emoji
		st.markdown(f"### The car was evaluated as: {readable_prediction} {emoji}")

		# Display the corresponding image for the prediction
		image_filename = image_mapping.get(prediction[0])
		st.image(image_filename, use_column_width=True)


elif model_choice == "Alcoholic Beverages Prediction":
	# Streamlit user interface
	st.markdown("# 🍻 Alcoholic Beverages Prediction")	
	st.write('Please enter the following liver blood test sample values:')

	# Collecting user inputs
	mcv = st.number_input('Mean Corpuscular Volume (typical values: 80-100 fL)', min_value=0)

	alkphos = st.number_input('Alkaline Phosphotase (typical values: 44-147 IU/L)', min_value=0)

	sgpt = st.number_input('Alanine Aminotransferase (typical values: 7-56 IU/L)', min_value=0)

	gammagt = st.number_input('Gamma-glutamyl Transpeptidase (typical values: 9-48 IU/L)', min_value=0)

	# Creating a dictionary for the input features
	input_data = {
		'mcv': mcv,
		'alkphos': alkphos,
		'sgpt': sgpt,
		# 'sgot': sgot, # This should be commented out or removed since we're not using it in the model
		'gammagt': gammagt,
		# 'drinks': drinks # This should be commented out or removed since it's the target
	}

	# Prediction button
	if st.button('Predict'):
		processed_data = preprocess_liver_input(input_data)
		prediction = liver_model.predict(processed_data)
		prediction_value = prediction.item()  # Convert single-value array to scalar
		prediction_text = f"{prediction_value:.1f}"  # Format with one decimal place
		st.write(f'The model estimates that this person consumes {prediction_text} half-pint equivalents per day')
