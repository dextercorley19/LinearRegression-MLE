import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt

def generate_sample_data(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Generates a random sample of linearly related data with a normally distributed error."""
    
    if 'x_sample' not in st.session_state or 'y_sample' not in st.session_state:
        x_sample = np.random.randn(size)
        y_sample = np.random.randint(1, 10) * x_sample + np.random.randn(size)
        st.session_state['x_sample'] = x_sample
        st.session_state['y_sample'] = y_sample
        random_index = np.random.randint(0, len(x_sample)-1, size=3)
        x_randoms = x_sample[random_index]
        y_randoms = y_sample[random_index]
        st.session_state['x_randoms'] = x_randoms
        st.session_state['y_randoms'] = y_randoms
    else:
        x_sample = st.session_state['x_sample']
        y_sample = st.session_state['y_sample']
        x_randoms = st.session_state['x_randoms']
        y_randoms = st.session_state['y_randoms']
    
    return x_sample, y_sample, x_randoms, y_randoms

def generate_ols_plot(x_randoms, y_randoms, B0, B1, sigma2):
    """Generates a plot showing the normal distribution for chosen input parameters and a random
    sample."""
    fig, ax = plt.subplots()

    # Define an orange, grey, and red color palette
    colors = ['#FFA500', '#808080', '#FF6347']  # Orange, gray, red

    # Plot normal distribution for the randomly selected point
    for i, x_random in enumerate(x_randoms):
        y_random = y_randoms[i]
        x_vals = np.linspace(y_random-20, y_random+20, 1000)
        y_vals = norm.pdf(x_vals, loc=(B0+B1*x_random), scale=np.sqrt(sigma2))
        
        # Plot the normal distribution with the corresponding color
        plt.plot(x_vals, y_vals, color=colors[i], label=f'Point {i+1}')
        
        # Highlight the randomly selected point with the same color
        ax.scatter([y_random], [y_vals.mean()], color=colors[i], s=100, zorder=5)

    # Add labels, legend, and title
    ax.set_title('Normal Distribution at 3 Randomly Selected Points')
    ax.set_xlabel('Y')
    ax.set_ylabel('Probability Density')
    ax.set_xbound(x_vals.min()-2, x_vals.max()+2)

    return fig, ax

