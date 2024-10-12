import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
from ordinary_least_squares import generate_sample_data, generate_ols_plot
from ridge_regression import (generate_data, fit_ridge_regression, plot_ridge_coefficients,
                              show_description, show_ridge_formula,
                                generate_likelihood_data, ridge_likelihood, plot_likelihood)
from lasso_regression import *
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


# OLS Parameter Estimation

# Page break
st.write('')

st.title('Estimating Paramters in Ordinary Least Squares')

# Generate and show x and Y data
x_sample, y_sample, x_randoms, y_randoms = generate_sample_data(10)

st.write('### Sample observations and responses')
st.write(f'x = {", ".join(str(round(x, 1)) for x in x_sample)}')
st.write(f'y = {", ".join(str(round(y, 1)) for y in y_sample)}')

st.write('### Methodology')

st.write('In OLS regression, we have 3 unknown parameters: B0, B1, and σ. \
         For each point in the data, we create a normal distribution with mean B0-B1Xi \
         and variance sima^2. This distribution represents the likelihood of obtaining our sample \
         response, yi for a given input xi with our chosen set of parameters B0, B1, and sigma. \
         We want to choose these parameters such that we maximize the probability of obtaining \
         our sample response for each data point.')

st.write('')

B0 = st.slider('Intercept (B0)', min_value=-20.0, max_value=20.0, value=0.0)
B1 = st.slider('Slope (B1)', min_value=-20.0, max_value=20.0, value=1.0)
sigma2 = st.slider('Variance (σ²)', min_value=0.1, max_value=50.0, value=1.0)

# Plot sample data

fig, ax = generate_ols_plot(x_randoms, y_randoms, B0, B1, sigma2)

st.pyplot(fig)

st.write('By combining each of these distributions into one likelihood function, we are able \
    to determine how well a set of coefficients fits the entire sample dataset. This overall \
        likelihood takes the following form:')

st.latex(r"""
    L(\beta_0, \beta_1, \sigma^2 | y, X) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi \sigma^2}} 
    \exp\left( -\frac{(y_i - \beta_0 - \beta_1 X_i)^2}{2\sigma^2} \right)
    """)

st.write('Which can be simplified to the following equation:')

st.latex(r"""
    L(\beta_0, \beta_1, \sigma^2 | y, X) = \left( \frac{1}{\sqrt{2\pi \sigma^2}} \right)^n 
    \exp\left( -\frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 X_i)^2 \right)
    """)

st.write('For the simple linear regression problem, this equation has a closed form solution which '
         'yields the regression coefficients that maximize likelihood across the entire sample dataset.')

# Ridge Regression

st.write('')

st.title('Estimating Paramters in Ridge Regression')

st.subheader("What is Ridge Regression?")
show_description()
show_ridge_formula()

X, y = generate_data(n_samples=100, n_features=5, noise=15)
alphas = st.slider("Choose range for λ (regularization strength)",
                    0.01, 100.0, (0.01, 50.0), step=0.5)
n_alphas = st.slider("Number of λ points", 10, 100, 50)

# Compute Ridge Regression Coefficients for multiple values of λ
alpha_values = np.linspace(alphas[0], alphas[1], n_alphas)
coefficients = np.zeros((n_alphas, X.shape[1]))

for i, alpha in enumerate(alpha_values):
    coef, intercept = fit_ridge_regression(X, y, alpha)
    coefficients[i, :] = coef

# Plot the coefficients as a function of λ
fig = plot_ridge_coefficients(alpha_values, coefficients)

st.plotly_chart(fig)

# Maximum likelihood estimate

st.subheader("Deriving Ridge Regression Coefficients")
X, y = generate_likelihood_data(n_samples=100, noise=5)

# Sliders for the Parameters
lambd = st.slider("λ (Regularization Strength)", 0.01, 10.0, 1.0, step=0.1)
beta_0 = st.slider("β₀ (Intercept)", -10.0, 10.0, 1.0, step=0.1)
beta_1 = st.slider("β₁ (Slope)", -10.0, 10.0, 2.0, step=0.1)

# Plot the Likelihood Function
st.subheader("Ridge Likelihood Contour Plot")
fig = plot_likelihood(X, y, lambd)
st.plotly_chart(fig)

# Calculate and Display Likelihood at the Slider Values
likelihood = ridge_likelihood(beta_0, beta_1, X, y, lambd)
st.write(f"Likelihood at β₀ = {beta_0}, β₁ = {beta_1}, λ = {lambd}: {likelihood:.2f}")

# Lasso Regression

st.write('')

st.title('Estimating Parameters in Lasso Regression')

st.subheader("What is Lasso Regression?")
show_description_lasso()  # Description for Lasso
show_lasso_formula()  # Formula for Lasso

# Generate data
X, y = generate_data_lasso(n_samples=100, n_features=5, noise=15)

# Sliders for Lasso regularization strength with unique keys
alphas = st.slider("Choose range for λ (regularization strength)",
                    0.01, 100.0, (0.01, 50.0), step=0.5, key="lasso_alpha_slider")
n_alphas = st.slider("Number of λ points", 10, 100, 50, key="lasso_n_alpha_slider")

# Compute Lasso Regression Coefficients for multiple values of λ
alpha_values = np.linspace(alphas[0], alphas[1], n_alphas)
coefficients = np.zeros((n_alphas, X.shape[1]))

for i, alpha in enumerate(alpha_values):
    coef, intercept = fit_lasso_regression(X, y, alpha)
    coefficients[i, :] = coef

# Plot the coefficients as a function of λ
fig = plot_lasso_coefficients(alpha_values, coefficients)
st.plotly_chart(fig)

# Maximum likelihood estimate for Lasso

st.subheader("Deriving Lasso Regression Coefficients")
X, y = generate_likelihood_data_lasso(n_samples=100, noise=5)

# Sliders for the Parameters with unique keys
lambd = st.slider("λ (Regularization Strength)", 0.01, 10.0, 1.0, step=0.1, key="lasso_lambd_slider")
beta_0 = st.slider("β₀ (Intercept)", -10.0, 10.0, 1.0, step=0.1, key="lasso_beta_0_slider")
beta_1 = st.slider("β₁ (Slope)", -10.0, 10.0, 2.0, step=0.1, key="lasso_beta_1_slider")

# Plot the Likelihood Function for Lasso
st.subheader("Lasso Likelihood Contour Plot")
fig = plot_likelihood_lasso(X, y, lambd)
st.plotly_chart(fig)

# Calculate and Display Likelihood at the Slider Values
likelihood = lasso_likelihood(beta_0, beta_1, X, y, lambd)
st.write(f"Likelihood at β₀ = {beta_0}, β₁ = {beta_1}, λ = {lambd}: {likelihood:.2f}")
