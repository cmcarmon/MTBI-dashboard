import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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