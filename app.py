import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from empath import Empath

# --- DATA LOADING & PREPARATION ---

@st.cache_data
def load_data():
    df = pd.read_csv('mbti_1.csv')
    # Clean text
    import re
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

# --- LIWC-STYLE ANALYSIS WITH EMPATH ---

@st.cache_data
def liwc_features_by_type(df, categories=None):
    lexicon = Empath()
    if categories is None:
        # Select a subset of relevant LIWC-like categories for demo
        categories = [
            'positive_emotion', 'negative_emotion', 'anger', 'sadness', 'joy',
            'work', 'money', 'social_media', 'family', 'friends', 'love', 'religion'
        ]
    results = {}
    for mbti_type in df['type'].unique():
        texts = df[df['type'] == mbti_type]['cleaned_posts']
        # Concatenate all posts for this type
        combined = " ".join(texts)
        features = lexicon.analyze(combined, categories=categories, normalize=True)
        results[mbti_type] = [features[cat] for cat in categories]
    heatmap_df = pd.DataFrame(results, index=categories).T
    return heatmap_df

liwc_df = liwc_features_by_type(df)

# --- STREAMLIT DASHBOARD ---

st.title('MBTI Essays Data Dashboard')

mbti_types = df['type'].unique()
selected_types = st.sidebar.multiselect('Select MBTI Types', mbti_types, default=mbti_types.tolist())
filtered_df = df[df['type'].isin(selected_types)]

st.write('### Type Distribution')
st.bar_chart(filtered_df['type'].value_counts())

st.write('### Sentiment by Type')
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x='type', y='sentiment', data=filtered_df, ax=ax)
plt.title('Sentiment by MBTI Type')
st.pyplot(fig)

st.write('### LIWC-Style Linguistic Features by MBTI Type')
# Filter heatmap to selected types
liwc_selected = liwc_df.loc[selected_types]
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.heatmap(liwc_selected, annot=True, cmap='coolwarm', linewidths=.5, ax=ax2)
plt.title('LIWC-Style Features (Empath) by MBTI Type')
plt.ylabel('MBTI Type')
plt.xlabel('LIWC/Empath Category')
st.pyplot(fig2)

st.write('### Sample Essays')
for _, row in filtered_df.head(5).iterrows():
    st.write(f"**{row['type']}**: {row['posts'][:300]}...")
