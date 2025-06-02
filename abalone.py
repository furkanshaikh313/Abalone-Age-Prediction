# app.py

import streamlit as st
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import requests
from io import BytesIO
import warnings
import time

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define a custom root_mean_squared_error function if it's not available
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

# Set Streamlit page configuration
st.set_page_config(
    page_title="🦪 Abalone Age Predictor",
    page_icon="🦪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced UI
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .title {
        color: #4B0082;
        text-align: center;
        font-size: 48px;
        font-weight: bold;
    }
    .subtitle {
        color: #483D8B;
        text-align: center;
        font-size: 24px;
    }
    .info {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        color: grey;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4B0082;
        margin: 10px 0;
    }
    .best-model {
        background-color: #e8f5e8;
        border-left: 5px solid #28a745;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Subtitle
st.markdown("<h1 class='title'>🦪 Abalone Age Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>Predict the Age of Abalone Using Multiple Machine Learning Models</h3>", unsafe_allow_html=True)

# Display Abalone Image with Reduced Size
image_url = "https://seahistory.org/wp-content/uploads/Abalone-Underwater.jpg"
response = requests.get(image_url)
if response.status_code == 200:
    abalone_image = Image.open(BytesIO(response.content))
    new_width = int(abalone_image.width * 0.35)
    new_height = int(abalone_image.height * 0.35)
    resized_image = abalone_image.resize((new_width, new_height))
    
    # Create three columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Center the image in the middle column
    with col2:
        st.image(resized_image, caption="Abalone Underwater")
else:
    st.error("Failed to load abalone image.")

# Add Description About Abalone and Project
st.markdown(
    """
    <div class="info">
    <h3>About Abalones</h3>
    <p>
    Abalones are marine snails known for their colorful, ear-shaped shells. They are harvested for their meat, which is considered a delicacy in many cultures. The age of an abalone is determined by counting the number of rings on its shell, a process that is both time-consuming and labor-intensive.
    </p>
    <h3>Project Overview</h3>
    <p>
    This enhanced project utilizes multiple machine learning algorithms including K-Nearest Neighbors, Random Forest, Support Vector Regression, Gradient Boosting, and Linear Models to predict the age of abalones based on various physical measurements. The application compares different models and helps identify the best performing algorithm for this specific dataset.
    </p>
    <p><b>
    Developed by Furkan Shaikh
    </p></b>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for user inputs and information
st.sidebar.header("🔧 Input Parameters")

def user_input_features():
    sex = st.sidebar.selectbox('Sex', ('M', 'F', 'I'))
    length = st.sidebar.slider('Length (mm)', 0.075, 0.815, 0.5, 0.01)
    diameter = st.sidebar.slider('Diameter (mm)', 0.055, 0.650, 0.5, 0.01)
    height = st.sidebar.slider('Height (mm)', 0.0, 1.130, 0.3, 0.01)
    whole_weight = st.sidebar.slider('Whole Weight (g)', 0.002, 2.826, 0.5, 0.01)
    shucked_weight = st.sidebar.slider('Shucked Weight (g)', 0.001, 1.488, 0.5, 0.01)
    viscera_weight = st.sidebar.slider('Viscera Weight (g)', 0.001, 0.760, 0.2, 0.01)
    shell_weight = st.sidebar.slider('Shell Weight (g)', 0.002, 1.005, 0.2, 0.01)
    data = {
        'Sex': sex,
        'Length': length,
        'Diameter': diameter,
        'Height': height,
        'Whole weight': whole_weight,
        'Shucked weight': shucked_weight,
        'Viscera weight': viscera_weight,
        'Shell weight': shell_weight
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Load the dataset using ucimlrepo
@st.cache_data
def load_data():
    abalone = fetch_ucirepo(id=1)  # ID 1 corresponds to the Abalone dataset
    X = abalone.data.features
    y = abalone.data.targets
    df = X.copy()
    df['Rings'] = y
    df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 
                  'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    return df

df = load_data()

# Sidebar: Model Selection
st.sidebar.subheader("🤖 Model Selection")
selected_models = st.sidebar.multiselect(
    "Choose models to compare:",
    ["K-Nearest Neighbors", "Random Forest", "Support Vector Regression", 
     "Gradient Boosting", "Linear Regression", "Ridge Regression", 
     "Lasso Regression", "Decision Tree"],
    default=["K-Nearest Neighbors", "Random Forest", "Gradient Boosting"]
)

# Sidebar: Show Dataset Information
st.sidebar.subheader("📚 Dataset Information")
dataset_info = """
**Title:** Abalone Data Set

**Sources:**
- **Original Owners:** Marine Resources Division, Marine Research Laboratories - Taroona, Department of Primary Industry and Fisheries, Tasmania.
- **Donor:** Sam Waugh, Department of Computer Science, University of Tasmania.
- **Date Received:** December 1995

**Number of Instances:** 4177  
**Number of Attributes:** 8  
**Class Distribution:** Rings range from 1 to 29.

**Description:**  
Predicting the age of abalone from physical measurements. The age of abalone is determined by counting the number of rings, which is a time-consuming task. This dataset includes various physical measurements to predict the age more efficiently.
"""
st.sidebar.info(dataset_info)

# Define model dictionary
def get_models():
    models = {
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=7),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Support Vector Regression": SVR(kernel='rbf', C=1.0, gamma='scale'),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10)
    }
    return models

# Main Content Area with Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Data Overview", "📈 Visualizations", "🤖 Model Comparison", "🏆 Best Model Analysis", "📊 Prediction"])

with tab1:
    st.subheader("🔍 Data Overview")
    if st.checkbox('Show Raw Data'):
        st.write(df.head())
    
    st.markdown("### 🧮 Statistical Summary")
    st.write(df.describe())
    
    st.markdown("### 📊 Class Distribution")
    fig_dist = px.histogram(df, x='Rings', nbins=30, title='Distribution of Rings', 
                            labels={'Rings':'Number of Rings'}, color_discrete_sequence=['#4B0082'])
    st.plotly_chart(fig_dist, use_container_width=True)

with tab2:
    st.subheader("📈 Exploratory Data Analysis")
    
    # Encode 'Sex' for numerical computations
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded['Sex'] = le.fit_transform(df_encoded['Sex'])
    
    # Correlation Heatmap
    st.markdown("#### 🔗 Correlation Heatmap")
    corr = df_encoded.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)
    
    # Scatter Plot: Length vs. Whole Weight
    st.markdown("#### 📏 Length vs. Whole Weight")
    fig_scatter = px.scatter(df, x='Length', y='Whole weight', color='Sex',
                             title='Whole Weight vs. Length by Sex',
                             labels={'Length':'Length (mm)', 'Whole weight':'Whole Weight (g)'})
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Box Plot: Height by Sex
    st.markdown("#### 📦 Height Distribution by Sex")
    fig_box = px.box(df, x='Sex', y='Height', title='Height Distribution by Sex',
                    labels={'Height':'Height (mm)'}, color='Sex', color_discrete_sequence=['#483D8B', '#6A5ACD', '#7B68EE'])
    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.subheader("🤖 Model Comparison and Training")
    
    if not selected_models:
        st.warning("Please select at least one model from the sidebar to proceed.")
    else:
        # Data Preprocessing
        X = df_encoded.drop('Rings', axis=1)
        y = df_encoded['Rings']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Scaling
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Get models
        all_models = get_models()
        models_to_compare = {name: model for name, model in all_models.items() if name in selected_models}
        
        # Train and evaluate models
        results = {}
        training_times = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(models_to_compare.items()):
            status_text.text(f'Training {name}...')
            
            # Record training time
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            end_time = time.time()
            training_times[name] = end_time - start_time
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = root_mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'CV_RMSE': cv_rmse,
                'Training_Time': training_times[name],
                'Model': model,
                'Predictions': y_pred
            }
            
            progress_bar.progress((i + 1) / len(models_to_compare))
        
        status_text.text('Training completed!')
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'RMSE': [results[name]['RMSE'] for name in results.keys()],
            'MAE': [results[name]['MAE'] for name in results.keys()],
            'R² Score': [results[name]['R²'] for name in results.keys()],
            'CV RMSE': [results[name]['CV_RMSE'] for name in results.keys()],
            'Training Time (s)': [results[name]['Training_Time'] for name in results.keys()]
        })
        
        # Display results table
        st.markdown("### 📊 Model Performance Comparison")
        st.dataframe(results_df.style.highlight_min(subset=['RMSE', 'MAE', 'CV RMSE', 'Training Time (s)']).highlight_max(subset=['R² Score']))
        
        # Find best model
        best_model_name = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
        best_model = results[best_model_name]['Model']
        
        st.success(f"🏆 Best performing model: **{best_model_name}** (Lowest RMSE: {results[best_model_name]['RMSE']:.3f})")
        
        # Visualize model comparison
        st.markdown("### 📈 Model Performance Visualizations")
        
        # RMSE Comparison
        fig_rmse = px.bar(results_df, x='Model', y='RMSE', 
                         title='RMSE Comparison Across Models',
                         color='RMSE', color_continuous_scale='Viridis')
        fig_rmse.update_xaxes(tickangle=45)
        st.plotly_chart(fig_rmse, use_container_width=True)
        
        # R² Score Comparison
        fig_r2 = px.bar(results_df, x='Model', y='R² Score', 
                       title='R² Score Comparison Across Models',
                       color='R² Score', color_continuous_scale='Viridis')
        fig_r2.update_xaxes(tickangle=45)
        st.plotly_chart(fig_r2, use_container_width=True)
        
        # Training Time Comparison
        fig_time = px.bar(results_df, x='Model', y='Training Time (s)', 
                         title='Training Time Comparison',
                         color='Training Time (s)', color_continuous_scale='Reds')
        fig_time.update_xaxes(tickangle=45)
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Actual vs Predicted for all models
        st.markdown("### 🔄 Actual vs Predicted Comparison")
        fig_comparison = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(results.keys())[:4],
            shared_xaxes=True, shared_yaxes=True
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (name, result) in enumerate(list(results.items())[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig_comparison.add_trace(
                go.Scatter(x=y_test, y=result['Predictions'], 
                          mode='markers', name=name,
                          marker=dict(color=colors[i % len(colors)], opacity=0.6)),
                row=row, col=col
            )
            
            # Add perfect prediction line
            min_val, max_val = min(y_test.min(), result['Predictions'].min()), max(y_test.max(), result['Predictions'].max())
            fig_comparison.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                          mode='lines', name='Perfect Prediction',
                          line=dict(dash='dash', color='red'),
                          showlegend=(i == 0)),
                row=row, col=col
            )
        
        fig_comparison.update_layout(height=800, showlegend=True, title_text="Actual vs Predicted - Model Comparison")
        fig_comparison.update_xaxes(title_text="Actual Rings")
        fig_comparison.update_yaxes(title_text="Predicted Rings")
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Store results for next tab
        st.session_state['results'] = results
        st.session_state['best_model_name'] = best_model_name
        st.session_state['best_model'] = best_model
        st.session_state['scaler'] = scaler
        st.session_state['le'] = le
        
        # Download results
        st.markdown("### 📥 Download Results")
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Model Comparison as CSV",
            data=csv,
            file_name='model_comparison_results.csv',
            mime='text/csv',
        )

with tab4:
    st.subheader("🏆 Best Model Detailed Analysis")
    
    if 'results' not in st.session_state:
        st.info("Please train models in the 'Model Comparison' tab first.")
    else:
        best_model_name = st.session_state['best_model_name']
        results = st.session_state['results']
        
        # Display best model metrics
        st.markdown(f"### 📊 {best_model_name} - Detailed Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card best-model">
                <h4>RMSE</h4>
                <h2>{results[best_model_name]['RMSE']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card best-model">
                <h4>MAE</h4>
                <h2>{results[best_model_name]['MAE']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card best-model">
                <h4>R² Score</h4>
                <h2>{results[best_model_name]['R²']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card best-model">
                <h4>CV RMSE</h4>
                <h2>{results[best_model_name]['CV_RMSE']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Best model specific analysis
        if best_model_name == "K-Nearest Neighbors":
            st.markdown("#### 🔍 K-Nearest Neighbors Hyperparameter Tuning")
            
            # K optimization (if KNN is the best model)
            X = df_encoded.drop('Rings', axis=1)
            y = df_encoded['Rings']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = st.session_state['scaler']
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            rmse_val = []
            k_list = list(range(1, 31))
            for k in k_list:
                model = KNeighborsRegressor(n_neighbors=k)
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
                rmse = root_mean_squared_error(y_test, pred)
                rmse_val.append(rmse)
            
            best_k = k_list[rmse_val.index(min(rmse_val))]
            
            fig_k_opt = px.line(x=k_list, y=rmse_val, markers=True,
                               title='RMSE vs Number of Neighbors (K)',
                               labels={'x': 'K', 'y': 'RMSE'})
            fig_k_opt.add_vline(x=best_k, line_dash="dash", line_color="red",
                               annotation_text=f'Optimal K = {best_k}')
            st.plotly_chart(fig_k_opt, use_container_width=True)
        
        elif best_model_name == "Random Forest":
            st.markdown("#### 🌲 Random Forest Feature Importance")
            
            # Feature importance for Random Forest
            feature_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 
                           'Shucked weight', 'Viscera weight', 'Shell weight']
            importances = results[best_model_name]['Model'].feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(importance_df, x='Feature', y='Importance',
                                  title='Feature Importance in Random Forest',
                                  color='Importance', color_continuous_scale='Viridis')
            fig_importance.update_xaxes(tickangle=45)
            st.plotly_chart(fig_importance, use_container_width=True)

with tab5:
    st.subheader("📊 Prediction")
    
    if 'best_model' not in st.session_state:
        st.info("Please train models in the 'Model Comparison' tab first.")
    else:
        st.markdown("Enter the abalone's characteristics in the sidebar to predict its age using the best performing model.")
        
        best_model = st.session_state['best_model']
        best_model_name = st.session_state['best_model_name']
        scaler = st.session_state['scaler']
        le = st.session_state['le']
        
        st.info(f"Using **{best_model_name}** for prediction (Best performing model)")
        
        # Encode and scale user input
        input_encoded = input_df.copy()
        input_encoded['Sex'] = le.transform(input_encoded['Sex'])
        input_scaled = scaler.transform(input_encoded)
        
        # Prediction
        prediction = best_model.predict(input_scaled)[0]
        age = prediction + 1.5  # As per dataset description
        
        # Display Prediction
        st.markdown("### 🎯 Predicted Age")
        st.success(f"The predicted age of the abalone is **{age:.1f} years** (Number of Rings: {prediction:.2f})")
        
        # Get predictions from all trained models for comparison
        if 'results' in st.session_state:
            st.markdown("### 🔄 Predictions from All Models")
            
            model_predictions = {}
            for name, result in st.session_state['results'].items():
                pred = result['Model'].predict(input_scaled)[0]
                model_predictions[name] = {
                    'Rings': pred,
                    'Age': pred + 1.5
                }
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame({
                'Model': list(model_predictions.keys()),
                'Predicted Rings': [model_predictions[name]['Rings'] for name in model_predictions.keys()],
                'Predicted Age (years)': [model_predictions[name]['Age'] for name in model_predictions.keys()]
            })
            
            st.dataframe(comparison_df.style.highlight_max(subset=['Predicted Rings', 'Predicted Age (years)']))
            
            # Visualization of predictions
            fig_pred_comparison = px.bar(comparison_df, x='Model', y='Predicted Age (years)',
                                       title='Age Predictions Across Different Models',
                                       color='Predicted Age (years)', 
                                       color_continuous_scale='Viridis',
                                       text='Predicted Age (years)')
            fig_pred_comparison.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_pred_comparison.update_xaxes(tickangle=45)
            st.plotly_chart(fig_pred_comparison, use_container_width=True)
        
        # Display input parameters
        st.markdown("### 📋 Input Parameters Summary")
        st.write(input_df)
        
        # Download Prediction
        st.markdown("### 📥 Download Prediction")
        prediction_df = pd.DataFrame({
            'Sex': input_df['Sex'],
            'Length (mm)': input_df['Length'],
            'Diameter (mm)': input_df['Diameter'],
            'Height (mm)': input_df['Height'],
            'Whole Weight (g)': input_df['Whole weight'],
            'Shucked Weight (g)': input_df['Shucked weight'],
            'Viscera Weight (g)': input_df['Viscera weight'],
            'Shell Weight (g)': input_df['Shell weight'],
            'Best Model': [best_model_name],
            'Predicted Rings': [prediction],
            'Predicted Age (years)': [age]
        })
        
        csv_pred = prediction_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prediction as CSV",
            data=csv_pred,
            file_name='abalone_prediction.csv',
            mime='text/csv',
        )

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
    <b>Enhanced Abalone Age Predictor with Multiple ML Models</b><br>
    <b>Developed by Furkan Shaikh</b><br> 
    Contact: [Furkan710284@gmail.com]
    </div>
    """, unsafe_allow_html=True)