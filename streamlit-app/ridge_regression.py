from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

# 1. Generate Sample Data
def generate_data(n_samples=100, n_features=1, noise=10):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    return X, y

# 2. Ridge Regression with Variable Lambda (Regularization Strength)
def fit_ridge_regression(X, y, alpha):
    ridge = Ridge(alpha=alpha)  # Alpha is the regularization parameter (lambda)
    ridge.fit(X, y)
    return ridge.coef_, ridge.intercept_

# 3. Plot Ridge Coefficients
def plot_ridge_coefficients(alphas, coefficients):
    fig = go.Figure()
    for i in range(coefficients.shape[1]):
        fig.add_trace(go.Scatter(x=alphas, y=coefficients[:, i], mode='lines+markers', name=f'Coefficient {i+1}'))

    fig.update_layout(
        title="Ridge Regression Coefficients vs Regularization Strength (λ)",
        xaxis_title="λ (Regularization Strength)",
        yaxis_title="Coefficient Value",
        template="plotly_dark"
    )
    return fig

def display_data(X, y):
    data = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(X.shape[1])])
    data['Target'] = y
    st.dataframe(data)

# 5. Show Ridge Regression Formula
def show_ridge_formula():
    st.latex(r"""
        \hat{\beta}_{\text{ridge}} = \arg \min_{\beta} \left\{ \sum_{i=1}^{n} (y_i - X_i \beta)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 \right\}
    """)

# 6. Show Description of Ridge Regression
def show_description():
    st.write("""
    **Ridge Regression** adds a regularization term (also called a penalty term) to the ordinary least squares (OLS) regression. 
    This penalty helps shrink the coefficients and reduces model complexity, making Ridge regression useful in cases of multicollinearity 
    or when there are more predictors than observations. The regularization term is controlled by the parameter lambda, 
    which adjusts the amount of shrinkage applied to the coefficients. As lambda increases, the model becomes more regularized, 
    and the coefficients shrink toward zero.
    """)

def generate_likelihood_data(n_samples=100, noise=10):
    np.random.seed(42)
    X = np.random.randn(n_samples)
    y = 2 * X + 1 + np.random.randn(n_samples) * noise  # y = 2*X + 1 + noise
    return X, y

# 2. Ridge Likelihood Function
def ridge_likelihood(beta_0, beta_1, X, y, lambd):
    residuals = y - (beta_0 + beta_1 * X)
    rss = np.sum(residuals**2)  # Residual sum of squares
    penalty = lambd * (beta_0**2 + beta_1**2)  # L2 regularization term
    return rss + penalty

# 3. Contour Plot of the Likelihood Function
def plot_likelihood(X, y, lambd):
    beta_0_vals = np.linspace(-10, 10, 100)
    beta_1_vals = np.linspace(-10, 10, 100)
    likelihood_vals = np.zeros((len(beta_0_vals), len(beta_1_vals)))

    for i, beta_0 in enumerate(beta_0_vals):
        for j, beta_1 in enumerate(beta_1_vals):
            likelihood_vals[i, j] = ridge_likelihood(beta_0, beta_1, X, y, lambd)

    beta_0_grid, beta_1_grid = np.meshgrid(beta_0_vals, beta_1_vals)

    fig = go.Figure(data=go.Contour(
        z=likelihood_vals,
        x=beta_0_vals,
        y=beta_1_vals,
        colorscale='Viridis',
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='white')
        ),
        colorbar=dict(title='Likelihood')
    ))

    fig.update_layout(
        title=f"Ridge Likelihood Contour Plot (λ = {lambd})",
        xaxis_title="β₀",
        yaxis_title="β₁",
        width=700,
        height=600
    )

    return fig