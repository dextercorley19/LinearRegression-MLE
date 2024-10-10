from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

def generate_data_lasso(n_samples=100, n_features=1, noise=10):
    """Generate sample data for regression fit."""
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    return X, y

def fit_lasso_regression(X, y, alpha):
    """Fit Lasso regression with variable lambda (alpha)"""
    lasso = Lasso(alpha=alpha)  # Alpha is the regularization parameter (lambda)
    lasso.fit(X, y)
    return lasso.coef_, lasso.intercept_

def plot_lasso_coefficients(alphas, coefficients):
    """Plot of Lasso coefficients"""
    fig = go.Figure()
    for i in range(coefficients.shape[1]):
        fig.add_trace(go.Scatter(x=alphas, y=coefficients[:, i], mode='lines+markers', name=f'Coefficient {i+1}'))

    fig.update_layout(
        title="Lasso Regression Coefficients vs Regularization Strength (λ)",
        xaxis_title="λ (Regularization Strength)",
        yaxis_title="Coefficient Value",
        template="plotly_dark"
    )
    return fig

def show_lasso_formula():
    """Display Lasso regression formula."""
    st.latex(r"""
        \hat{\beta}_{\text{lasso}} = \arg \min_{\beta} \left\{ \sum_{i=1}^{n} (y_i - X_i \beta)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right\}
    """)

def show_description_lasso():
    """Describe Lasso regression."""
    st.write("""
    **Lasso Regression** (Least Absolute Shrinkage and Selection Operator) adds an L1 regularization term to the ordinary least squares (OLS) regression.
    The L1 penalty forces some of the coefficient estimates to be exactly zero, thus performing feature selection. As the λ parameter increases, 
    more coefficients are shrunk to zero. This makes Lasso particularly useful when you have a large number of features, as it helps in reducing the complexity
    of the model by selecting a subset of features.
    """)

def generate_likelihood_data_lasso(n_samples=100, noise=10):
    """Generate data for likelihood derivation."""
    np.random.seed(42)
    X = np.random.randn(n_samples)
    y = 2 * X + 1 + np.random.randn(n_samples) * noise  # y = 2*X + 1 + noise
    return X, y

def lasso_likelihood(beta_0, beta_1, X, y, lambd):
    """Compute the likelihood function for Lasso regression"""
    residuals = y - (beta_0 + beta_1 * X)
    rss = np.sum(residuals**2)  # Residual sum of squares
    penalty = lambd * (np.abs(beta_0) + np.abs(beta_1))  # L1 regularization term
    return rss + penalty

def plot_likelihood_lasso(X, y, lambd):
    """Generate contour plot of likelihood."""
    beta_0_vals = np.linspace(-10, 10, 100)
    beta_1_vals = np.linspace(-10, 10, 100)
    likelihood_vals = np.zeros((len(beta_0_vals), len(beta_1_vals)))

    for i, beta_0 in enumerate(beta_0_vals):
        for j, beta_1 in enumerate(beta_1_vals):
            likelihood_vals[i, j] = lasso_likelihood(beta_0, beta_1, X, y, lambd)

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
        title=f"Lasso Likelihood Contour Plot (λ = {lambd})",
        xaxis_title="β₀",
        yaxis_title="β₁",
        width=700,
        height=600
    )

    return fig
