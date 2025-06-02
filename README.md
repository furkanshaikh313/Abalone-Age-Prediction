# 🦪 Abalone Age Predictor

This project presents an interactive web application built with Streamlit that predicts the age of abalones using various machine learning regression models. The application allows users to input physical measurements of an abalone and instantly receive a predicted age, along with detailed model performance comparisons and visualizations.

---

## 🌟 Features

* **Interactive Input:** Easily adjust abalone physical characteristics (length, diameter, weight, etc.) using sliders and dropdowns in the sidebar.
* **Comprehensive Data Overview:** Explore the dataset's raw data, statistical summaries, and age distribution.
* **In-depth Exploratory Data Analysis (EDA):** Visualize correlations, relationships between features, and distributions with interactive plots.
* **Multiple Model Comparison:** Train and compare the performance of several regression algorithms, including:
    * K-Nearest Neighbors
    * Random Forest
    * Support Vector Regression
    * Gradient Boosting
    * Linear Regression
    * Ridge Regression
    * Lasso Regression
    * Decision Tree
* **Performance Metrics:** Evaluate models based on RMSE, MAE, R² Score, and Cross-Validation RMSE.
* **Best Model Identification:** Automatically identifies and highlights the best-performing model based on the lowest RMSE.
* **Detailed Best Model Analysis:** Provides specific insights and visualizations for the top-performing model, such as hyperparameter tuning for KNN or feature importance for Random Forest.
* **Real-time Predictions:** Get instant age predictions for user-defined abalone attributes using the best-performing model.
* **Downloadable Results:** Download model comparison and individual prediction results as CSV files.
* **Customizable UI:** Enhanced user interface with custom CSS for a clean and engaging experience.

---

## 🌐 Dataset

The application utilizes the **Abalone Data Set** from the UCI Machine Learning Repository.

* **Title:** Abalone Data Set
* **Sources:** Marine Resources Division, Marine Research Laboratories - Taroona, Department of Primary Industry and Fisheries, Tasmania.
* **Donor:** Sam Waugh, Department of Computer Science, University of Tasmania.
* **Date Received:** December 1995
* **Number of Instances:** 4177
* **Number of Attributes:** 8 (plus the target variable, Rings)
* **Description:** The dataset contains various physical measurements of abalones, with the goal of predicting their age (number of rings). The age of abalone is typically determined by counting the number of rings, a labor-intensive process. This project aims to automate this prediction using machine learning. The actual age is `Rings + 1.5` years, as the dataset describes.

---

## 🛠️ Installation

To run this application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
    (Replace `<repository_url>` and `<repository_directory>` with the actual values.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    (Assuming you have a `requirements.txt` file. If not, create one with the following contents):
    ```
    streamlit
    pandas
    numpy
    ucimlrepo
    scikit-learn
    matplotlib
    seaborn
    plotly
    Pillow
    requests
    ```

---

## 🚀 Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  The application will open in your web browser. You can then navigate through the tabs:
    * **🔍 Data Overview:** View dataset details and distributions.
    * **📈 Visualizations:** Explore relationships between features.
    * **🤖 Model Comparison:** Select models to train and compare their performance.
    * **🏆 Best Model Analysis:** Dive deeper into the metrics and insights of the top-performing model.
    * **📊 Prediction:** Input abalone characteristics and get an age prediction using the best model.

---

## 💡 How it Works

The application performs the following steps:

1.  **Data Loading:** The Abalone dataset is fetched directly from the UCI Machine Learning Repository.
2.  **Preprocessing:**
    * The 'Sex' column (categorical) is label encoded.
    * Numerical features are scaled using `MinMaxScaler` to ensure consistent ranges for model training.
3.  **Model Training and Evaluation:**
    * The dataset is split into training and testing sets.
    * Selected machine learning models are trained on the scaled training data.
    * Each model's performance is evaluated using metrics like RMSE, MAE, R² Score, and Cross-Validation RMSE.
4.  **Prediction:**
    * User input features are encoded and scaled using the same preprocessors fitted on the training data.
    * The best-performing model makes a prediction based on the user's input.
    * The predicted number of rings is converted to age by adding 1.5.

---

## 🧑‍💻 Developed By

**Furkan Shaikh**
Contact: Furkan710284@gmail.com