import streamlit as st
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import seaborn as sns

# --- Load and Train the Model ---


@st.cache_data
def load_data():
    df = pd.read_csv("dataset file string")

    # Total engagement = likes + retweets + replies
    df['total_engagement'] = df['likeCount'] + \
        df['retweetCount'] + df['replyCount']
    # Combined = total + views (label only!)
    df['engagement_combined'] = df['total_engagement'] + \
        df['viewCount'].fillna(0)

    # Log transform the target variable to handle skewness
    df['log_engagement'] = np.log1p(df['engagement_combined'])

    # Features
    df['createdAt'] = pd.to_datetime(
        df['createdAt'], errors='coerce', utc=True)
    df['created_hour'] = df['createdAt'].dt.tz_convert(
        "America/New_York").dt.hour.fillna(12)
    df['created_day'] = df['createdAt'].dt.tz_convert(
        "America/New_York").dt.dayofweek
    df['tweet_length'] = df['text'].astype(str).apply(len)
    df['has_link'] = df['text'].astype(str).apply(
        lambda x: 1 if re.search(r"http[s]?://", x) else 0)
    df['hashtag_count'] = df['text'].astype(str).apply(
        lambda x: len(re.findall(r"#\w+", x)))
    df['mention_count'] = df['text'].astype(str).apply(
        lambda x: len(re.findall(r"@\w+", x)))

    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['text'].astype(str).apply(
        lambda x: analyzer.polarity_scores(x)["compound"])
    df['sentiment_pos'] = df['text'].astype(str).apply(
        lambda x: analyzer.polarity_scores(x)["pos"])
    df['sentiment_neg'] = df['text'].astype(str).apply(
        lambda x: analyzer.polarity_scores(x)["neg"])

    def categorize_tweet_type(text):
        text = str(text).lower()
        if any(word in text for word in ["announces", "announcement", "launch", "new"]):
            return 1
        elif any(word in text for word in ["wins", "becomes", "breaks", "first"]):
            return 2
        elif any(word in text for word in ["thanks", "celebrates", "congrats", "happy"]):
            return 3
        elif any(word in text for word in ["fan", "supporters", "community"]):
            return 4
        elif any(word in text for word in ["anniversary", "anniversaries", "years since"]):
            return 5
        else:
            return 0

    df['tweet_type'] = df['text'].apply(categorize_tweet_type)

    # Remove extreme outliers (over 3 standard deviations from the mean in log space)
    log_mean = df['log_engagement'].mean()
    log_std = df['log_engagement'].std()
    df = df[(df['log_engagement'] <= log_mean + 3*log_std) &
            (df['log_engagement'] >= log_mean - 3*log_std)]

    return df, analyzer


def train_and_evaluate_model(df, target_col='log_engagement'):
    # Model features
    features = [
        'tweet_length', 'has_link', 'created_hour', 'created_day', 'tweet_type',
        'hashtag_count', 'mention_count', 'sentiment', 'sentiment_pos', 'sentiment_neg'
    ]

    X = df[features]
    y = df[target_col]

    # Check for correlation between features
    corr_matrix = X.corr().abs()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # If we used log transformation, convert back for evaluation
    if target_col == 'log_engagement':
        y_test_original = np.expm1(y_test)
        y_pred_original = np.expm1(y_pred)
    else:
        y_test_original = y_test
        y_pred_original = y_pred

    # Calculate accuracy metrics
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    r2 = r2_score(y_test, y_pred)  # Calculate R² on transformed scale

    # Cross-validation score
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

    # Calculate custom percentage accuracy for engagement
    percent_error = np.abs(
        (y_test_original - y_pred_original) / (y_test_original + 1)) * 100
    within_20_percent = np.mean(percent_error <= 20)

    # Get feature importances
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': pipeline.named_steps['model'].feature_importances_
    }).sort_values(by='Importance', ascending=False)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'CV_R²_Mean': cv_scores.mean(),
        'CV_R²_Std': cv_scores.std(),
        'Within_20%': within_20_percent
    }

    return pipeline, metrics, X_test, y_test, y_pred, y_test_original, y_pred_original, feature_importance, corr_matrix


# --- Main Streamlit App ---
st.set_page_config(page_title="Tweet Engagement Predictor", layout="wide")
st.title("📊 Tweet Engagement Prediction and Analysis")

# Load data
df, analyzer = load_data()

# Train model
with st.spinner("Training model... This may take a moment"):
    pipeline, metrics, X_test, y_test, y_pred, y_test_original, y_pred_original, feature_importance, corr_matrix = train_and_evaluate_model(
        df)

# Create tabs
tab1, tab2, tab3 = st.tabs(
    ["Make Predictions", "Model Accuracy", "Data Analysis"])

with tab1:
    st.header("Predict Total Engagement")

    tweet_text = st.text_area("📝 Paste your tweet idea", height=150)

    col1, col2 = st.columns(2)

    with col1:
        time_of_day = st.selectbox("🕐 Select Posting Time", [
            "Morning (5 AM - 12 PM)",
            "Afternoon (12 PM - 5 PM)",
            "Evening (5 PM - 10 PM)"
        ])

        tweet_type = st.selectbox("🏷️ Select Tweet Type", [
            "Announcement", "Breaking News", "Celebration", "Fan Moment", "Anniversary", "Other"
        ])

    with col2:
        day_of_week = st.selectbox("📅 Day of Week", [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ])

        sentiment_analysis = analyzer.polarity_scores(tweet_text)
        st.metric(
            "Sentiment Score",
            f"{sentiment_analysis['compound']:.2f}",
            delta=None,
            delta_color="normal"
        )

    # --- Feature Engineering ---
    def get_time_hour(slot):
        return {"Morning (5 AM - 12 PM)": 9, "Afternoon (12 PM - 5 PM)": 14, "Evening (5 PM - 10 PM)": 19}[slot]

    def get_day_number(day):
        return {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}[day]

    def get_type_code(label):
        return {"Announcement": 1, "Breaking News": 2, "Celebration": 3,
                "Fan Moment": 4, "Anniversary": 5, "Other": 0}.get(label, 0)

    def contains_link(text):
        return 1 if re.search(r"http[s]?://", str(text)) else 0

    def extract_hashtag_count(text):
        return len(re.findall(r"#\w+", str(text)))

    def extract_mention_count(text):
        return len(re.findall(r"@\w+", str(text)))

    # --- Predict ---
    if st.button("🔮 Predict Total Engagement"):
        tweet_length = len(tweet_text)
        has_link = contains_link(tweet_text)
        created_hour = get_time_hour(time_of_day)
        created_day = get_day_number(day_of_week)
        tweet_type_code = get_type_code(tweet_type)
        hashtag_count = extract_hashtag_count(tweet_text)
        mention_count = extract_mention_count(tweet_text)
        sentiment_scores = analyzer.polarity_scores(tweet_text)

        input_df = pd.DataFrame([{
            'tweet_length': tweet_length,
            'has_link': has_link,
            'created_hour': created_hour,
            'created_day': created_day,
            'tweet_type': tweet_type_code,
            'hashtag_count': hashtag_count,
            'mention_count': mention_count,
            'sentiment': sentiment_scores["compound"],
            'sentiment_pos': sentiment_scores["pos"],
            'sentiment_neg': sentiment_scores["neg"]
        }])

        # Predict (in log space)
        log_prediction = pipeline.predict(input_df)[0]
        # Convert back from log space
        predicted_engagement = int(np.expm1(log_prediction))

        st.subheader("🎯 Predicted Total Engagement:")
        st.success(
            f"{predicted_engagement:,} (includes views + likes + replies + retweets)")

        if predicted_engagement > 5_000_000:
            st.balloons()

        # Suggest improvements based on feature importance
        st.subheader("Suggestions to Improve Engagement")
        top_features = feature_importance.head(3)['Feature'].tolist()

        suggestions = []

        if 'tweet_length' in top_features:
            avg_length = df['tweet_length'].mean()
            if tweet_length < avg_length:
                suggestions.append(
                    f"🔄 Consider making your tweet longer (current: {tweet_length} chars, average: {int(avg_length)} chars)")
            elif tweet_length > avg_length * 1.5:
                suggestions.append(
                    f"🔄 Consider making your tweet more concise (current: {tweet_length} chars, average: {int(avg_length)} chars)")

        if 'hashtag_count' in top_features:
            optimal_hashtags = df.groupby('hashtag_count')[
                'engagement_combined'].mean().sort_values(ascending=False).index[0]
            if hashtag_count < optimal_hashtags:
                suggestions.append(
                    f"#️⃣ Add more hashtags (optimal number: {optimal_hashtags})")
            elif hashtag_count > optimal_hashtags:
                suggestions.append(
                    f"#️⃣ Use fewer hashtags (optimal number: {optimal_hashtags})")

        if 'created_hour' in top_features:
            best_hour = df.groupby('created_hour')['engagement_combined'].mean(
            ).sort_values(ascending=False).index[0]
            best_time = "Morning" if 5 <= best_hour < 12 else "Afternoon" if 12 <= best_hour < 17 else "Evening"
            if get_time_hour(time_of_day) != best_hour:
                suggestions.append(
                    f"🕒 Consider posting during {best_time} for higher engagement")

        if 'sentiment' in top_features:
            best_sentiment = df.loc[df['engagement_combined'].idxmax(
            )]['sentiment']
            if abs(sentiment_scores["compound"] - best_sentiment) > 0.5:
                if best_sentiment > 0:
                    suggestions.append(
                        "😊 Try using more positive language in your tweet")
                else:
                    suggestions.append(
                        "🤔 Consider using more neutral or factual language")

        if suggestions:
            for suggestion in suggestions:
                st.markdown(suggestion)
        else:
            st.markdown(
                "✅ Your tweet already follows the best practices based on our model!")

with tab2:
    st.header("Model Performance Metrics")

    # Display metrics in a nice format
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("R² Score", f"{metrics['R²']:.4f}")
        st.metric("Mean Absolute Error", f"{metrics['MAE']:,.2f}")

    with col2:
        st.metric("Cross-Validation R²",
                  f"{metrics['CV_R²_Mean']:.4f} ± {metrics['CV_R²_Std']:.4f}")
        st.metric("Root Mean Squared Error", f"{metrics['RMSE']:,.2f}")

    with col3:
        st.metric("Predictions Within 20%",
                  f"{metrics['Within_20%']*100:.1f}%")

    # Add explanation if R2 is negative
    if metrics['R²'] < 0:
        st.warning("""
        ### Note on Negative R² Score:
        
        A negative R² indicates the model is not fitting the data well. This could be due to:
        
        1. **High variability in social media engagement** - Engagement can be highly unpredictable
        2. **Missing important features** - There may be factors influencing engagement that aren't captured in our model
        3. **Outliers in the data** - Some tweets may have unusually high/low engagement
        
        See the Data Analysis tab for insights and suggestions on improving the model.
        """)

    # Add visual comparison of actual vs predicted
    st.subheader("Actual vs Predicted Engagement")

    # Create a scatter plot of actual vs predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test_original, y_pred_original, alpha=0.5)

    # Add perfect prediction line
    max_val = max(y_test_original.max(), y_pred_original.max())
    ax.plot([0, max_val], [0, max_val], 'r--')

    ax.set_xlabel('Actual Engagement')
    ax.set_ylabel('Predicted Engagement')
    ax.set_title('Actual vs Predicted Engagement')

    # Use log scale for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')

    st.pyplot(fig)

    # Feature importance
    st.subheader("Feature Importance")

    # Create a horizontal bar chart of feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance.plot(kind='barh', x='Feature', y='Importance', ax=ax)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')

    st.pyplot(fig)

    # Explanation of metrics
    st.subheader("Understanding the Metrics")
    st.markdown("""
    - **R² Score**: Ranges from negative infinity to 1. A value of 1 indicates perfect prediction, 0 means predictions are no better than using the mean value, and negative values indicate the model performs worse than predicting the mean.
    
    - **Cross-Validation R²**: The R² score calculated using 5-fold cross-validation, which helps assess how well the model generalizes to new data.
    
    - **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual engagement counts. Lower is better.
    
    - **Root Mean Squared Error (RMSE)**: Similar to MAE but penalizes large errors more heavily. Lower is better.
    
    - **Within 20%**: Percentage of predictions that are within 20% of the actual value, a practical measure of accuracy.
    """)

with tab3:
    st.header("Data Analysis & Insights")

    # Basic dataset statistics
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tweets", f"{len(df):,}")
    with col2:
        st.metric("Avg. Engagement",
                  f"{df['engagement_combined'].mean():,.1f}")
    with col3:
        st.metric("Max Engagement", f"{df['engagement_combined'].max():,}")
    with col4:
        st.metric("Median Engagement",
                  f"{df['engagement_combined'].median():,.1f}")

    # Distribution of engagement
    st.subheader("Engagement Distribution")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(np.log1p(df['engagement_combined']), bins=30, kde=True, ax=ax)
    ax.set_xlabel('Log Engagement')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Log-Transformed Engagement')
    st.pyplot(fig)

    # Time analysis
    st.subheader("Engagement by Time of Day")

    hour_engagement = df.groupby('created_hour')[
        'engagement_combined'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='created_hour', y='engagement_combined',
                data=hour_engagement, ax=ax)
    ax.set_xlabel('Hour of Day (EST)')
    ax.set_ylabel('Average Engagement')
    ax.set_title('Average Engagement by Hour of Day')
    st.pyplot(fig)

    # Day of week analysis
    st.subheader("Engagement by Day of Week")

    day_engagement = df.groupby('created_day')[
        'engagement_combined'].mean().reset_index()
    day_names = ['Monday', 'Tuesday', 'Wednesday',
                 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_engagement['day_name'] = day_engagement['created_day'].apply(
        lambda x: day_names[int(x)])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='day_name', y='engagement_combined',
                data=day_engagement, ax=ax)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Average Engagement')
    ax.set_title('Average Engagement by Day of Week')
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Feature Correlation")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Between Features')
    st.pyplot(fig)

    