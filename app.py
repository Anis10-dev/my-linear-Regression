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
        layout="centered",  # Center the layout for better scaling on mobile
        initial_sidebar_state="collapsed",  # Sidebar collapsed by default
    )

    # Inject custom CSS for responsiveness
    st.markdown(
        """
        <style>
            /* Make sliders and buttons more touch-friendly */
            .stSlider > div {
                height: auto !important;
            }
            /* Adjust font size for mobile */
            body {
                font-size: 16px;
            }
            /* Improve spacing */
            .block-container {
                padding: 1rem 1rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar for input
    st.sidebar.header('Input Features')
    st.sidebar.write("Enter or adjust the values for prediction:")

    # Use smaller input fields for mobile
    avg_session_length = st.sidebar.number_input('Avg. Session Length', 29.532429, 36.139662, 33.053194)
    time_on_app = st.sidebar.number_input('Time on App', 8.508152, 15.126994, 12.052488)
    time_on_website = st.sidebar.number_input('Time on Website', 33.913847, 40.005182, 37.060445)
    length_of_membership = st.sidebar.number_input('Length of Membership', 0.269901, 6.922689, 3.533462)

    # Main panel
    st.title('💸 Yearly Amount Spent Prediction App 💸')

    st.markdown("""
    Welcome to the **Yearly Amount Spent Prediction App**! Enter values for the features on the sidebar and click **Predict** to see the result.
    """)

    # Prepare input data
    input_data = np.array([[avg_session_length, time_on_app, time_on_website, length_of_membership]])

    # Normalize the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict button
    if st.button('Predict'):
        prediction = model_yearly_spent.predict(input_data_scaled)
        st.success(f'Predicted Yearly Amount Spent: **${prediction[0]:.2f}**')

if __name__ == '__main__':
    main()
