import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Streamlit app
st.title("Introduction to Machine Learning with Probability")

# Generate synthetic data for regression
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Feature
y = 2 * X + 1 + np.random.randn(100, 1) * 2  # Linear relationship with noise

# Sidebar for user input
st.sidebar.header("Settings")
noise_level = st.sidebar.slider("Noise Level", min_value=0.1, max_value=5.0, value=2.0)
random_seed = st.sidebar.number_input("Random Seed", value=42)

# Update synthetic data based on user input
np.random.seed(random_seed)
y = 2 * X + 1 + np.random.randn(100, 1) * noise_level  # Linear relationship with noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

# Display results
st.write(f"Mean Squared Error: {mse:.2f}")

# Visualize the regression line
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, label='True values')
ax.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
ax.set_xlabel('Feature')
ax.set_ylabel('Target')
ax.legend()

# Show the plot in Streamlit
st.pyplot(fig)
