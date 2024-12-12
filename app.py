import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model_yearly_spent = joblib.load('linearmodel.pkl')
scaler = joblib.load('scaler.pkl')

# Define the Streamlit app
def main():
    st.set_page_config(
        page_title="Yearly Amount Spent Prediction",
        page_icon="💸",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar
    st.sidebar.header('Input Features')
    st.sidebar.write("Adjust the values to get the prediction.")

    # Input fields for user data in the sidebar
    avg_session_length = st.sidebar.slider('Avg. Session Length', 29.532429, 36.139662, 33.053194)
    time_on_app = st.sidebar.slider('Time on App', 8.508152, 15.126994, 12.052488)
    time_on_website = st.sidebar.slider('Time on Website', 33.913847, 40.005182, 37.060445)
    length_of_membership = st.sidebar.slider('Length of Membership', 0.269901, 6.922689, 3.533462)

    # Main panel
    st.title('💸 Yearly Amount Spent Prediction App 💸')

    st.markdown("""
    Welcome to the **Yearly Amount Spent Prediction App**! Adjust the sliders on the sidebar to set the values for the features, and click the **Predict** button to see the predicted yearly amount spent.
    """)

    # Create input array
    input_data = np.array([[avg_session_length, time_on_app, time_on_website, length_of_membership]])

    # Normalize the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    if st.button('Predict'):
        prediction = model_yearly_spent.predict(input_data_scaled)
        st.success(f'Predicted Yearly Amount Spent: **${prediction[0]:.2f}**')

    # Display a fancy image
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQcKzwOSYVfFk28-BBqwg1TeqnKxGae0Njspw&s', use_container_width =True)

if __name__ == '__main__':
    main()
