import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm

# Title of the app
st.title("MLE Simulation for Normal Distribution")

# Given data
Y = np.array([78.0, 95.5, 100.3, 100.6, 102.8, 107.8, 109.1, 110.8, 113.9, 125.0])
n = len(Y)

# Display the data horizontally
st.write("### Observations from Normal Distribution")
st.write("Y =", ', '.join(f"{value:.1f}" for value in Y))  # Format and join the array


# Sliders for Mean (μ) and Variance (σ²)
mu_slider = st.slider("Mean (μ)", min_value=float(Y.min()), max_value=float(Y.max()), value=float(Y.mean()), step=0.1)
sigma_slider = st.slider("Variance (σ²)", min_value=0.1, max_value=50.0, value=float(Y.var()), step=0.1)

# Calculate log-likelihood
def log_likelihood(mu, sigma2, data):
    return np.sum(norm.logpdf(data, loc=mu, scale=np.sqrt(sigma2)))

# Calculate the log-likelihood for the selected parameters
log_likelihood_value = log_likelihood(mu_slider, sigma_slider, Y)

# Display the log-likelihood value
st.write("### Log-Likelihood Value")
st.write(f"ℓ(μ, σ²) = {log_likelihood_value:.2f}")

# Plotting the normal distribution based on the selected parameters
x = np.linspace(Y.min() - 10, Y.max() + 10, 1000)
pdf = norm.pdf(x, loc=mu_slider, scale=np.sqrt(sigma_slider))

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, label=f'Normal Distribution\nμ = {mu_slider:.2f}, σ² = {sigma_slider:.2f}', color='blue')
plt.hist(Y, density=True, bins=10, alpha=0.6, color='gray', label='Data Histogram')
plt.title('Normal Distribution Fit')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
st.pyplot(plt)
