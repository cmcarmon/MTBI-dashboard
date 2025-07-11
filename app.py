import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder

# --- DATA LOADING & PREPARATION ---

# Load the MBTI dataset (make sure mbti_1.csv is in the same folder)
@st.cache_data
def load_data():
    df = pd.read_csv('mbti_1.csv')
    # Clean text
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text.lower()
    df['cleaned_posts'] = df['posts'].apply(clean_text)
    # Encode MBTI types
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    # Feature engineering
    df['word_count'] = df['cleaned_posts'].apply(lambda x: len(x.split()))
    df['sentiment'] = df['cleaned_posts'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

df = load_data()

# --- STREAMLIT DASHBOARD ---

st.title('MBTI Essays Data Dashboard')

mbti_types = df['type'].unique()
selected_types = st.sidebar.multiselect('Select MBTI Types', mbti_types, default=mbti_types.tolist())
filtered_df = df[df['type'].isin(selected_types)]

st.write('### Type Distribution')
st.bar_chart(filtered_df['type'].value_counts())

st.write('### Sentiment by Type')
# Option 1: Seaborn/Matplotlib
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x='type', y='sentiment', data=filtered_df, ax=ax)
plt.title('Sentiment by MBTI Type')
st.pyplot(fig)

# Option 2: Plotly (interactive)
# fig = px.box(filtered_df, x='type', y='sentiment', title='Sentiment by MBTI Type')
# st.plotly_chart(fig)

st.write('### Sample Essays')
for _, row in filtered_df.head(5).iterrows():
    st.write(f"**{row['type']}**: {row['posts'][:300]}...")
