## Maximum Likelihood Estimators

### An exploration of MLE for OLS, Ridge, and Lasso Linear Regression

#### This guide will walk you through how to run the Streamlit app using Docker.

##### Prerequisites

Before getting started, ensure that you have the following installed:

- Docker
- Git

##### Clone the Repository

First, clone the GitHub repository containing the Streamlit app:

```bash
cd path/to/destination/dir
git clone https://github.com/dextercorley19/LinearRegression-MLE
```

##### Build the Docker Image

Next, build the Docker image using the Dockerfile provided in the repository:

```bash
cd streamlit-app
docker compose up --build
```

##### Access the App

After running the container, open your web browser and go to:

http://localhost:8501


##### Stopping the Container

To stop the running container, press CTRL+C in the terminal where the container is running.

##### To remove the container and image after use, you can run:

```bash
docker rm <container-id>
docker rmi streamlit-app
```
