import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Load the data
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    return movies

movies = load_data()

# Data Preprocessing
def parse_list(x):
    if isinstance(x, str):
        try:
            return ' '.join([i['name'] for i in ast.literal_eval(x)])
        except:
            return ''
    return ''

movies['genres'] = movies['genres'].apply(parse_list)
movies['keywords'] = movies['keywords'].apply(parse_list)
movies['production_companies'] = movies['production_companies'].apply(parse_list)

def create_soup(x):
    return (x['genres'] + ' ' + x['keywords'] + ' ' +
            x['production_companies'] + ' ' + x['original_language'])

movies['soup'] = movies.apply(create_soup, axis=1)

# TF-IDF Vectorization
@st.cache_resource
def create_similarity_matrix():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = create_similarity_matrix()

# Create a reverse mapping of titles and DataFrame indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Content-based Recommendation Function
def content_based_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Clustering
@st.cache_resource
def perform_clustering():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['soup'])
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(tfidf_matrix)
    return kmeans.labels_

movies['cluster'] = perform_clustering()

# Function to get recommendations
def get_recommendations(title):
    if title not in movies['title'].values:
        return "Movie not found in the database. Please try again."

    content_recs = content_based_recommendations(title)

    # Get the cluster of the input movie
    movie_cluster = movies[movies['title'] == title]['cluster'].values[0]

    # Get movies from the same cluster
    cluster_recs = movies[movies['cluster'] == movie_cluster]['title'].sample(5)

    return {
        'Content-based Recommendations': content_recs.tolist(),
        'Cluster-based Recommendations': cluster_recs.tolist()
    }

# Streamlit app
st.title('Movie Recommendation System')

# Sidebar
st.sidebar.header('Navigation')
page = st.sidebar.radio('Go to', ['Home', 'Get Recommendations', 'Top Movies', 'Visualizations'])

if page == 'Home':
    st.write('Welcome to the Movie Recommendation System!')
    st.write('Use the sidebar to navigate through different sections of the app.')

elif page == 'Get Recommendations':
    st.header('Get Movie Recommendations')
    movie_title = st.text_input('Enter a movie title:')
    if st.button('Get Recommendations'):
        if movie_title:
            recommendations = get_recommendations(movie_title)
            if isinstance(recommendations, str):
                st.write(recommendations)
            else:
                st.subheader('Content-based Recommendations:')
                st.write(recommendations['Content-based Recommendations'])
                st.subheader('Cluster-based Recommendations:')
                st.write(recommendations['Cluster-based Recommendations'])
        else:
            st.write('Please enter a movie title.')

elif page == 'Top Movies':
    st.header('Top Movies by Vote Average')
    top_movies = movies[movies['vote_count'] > 1000].sort_values('vote_average', ascending=False).head(10)
    st.dataframe(top_movies[['title', 'vote_average', 'vote_count']])

elif page == 'Visualizations':
    st.header('Movie Dataset Visualizations')
    
    st.subheader('Distribution of Movies Across Clusters')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='cluster', data=movies, ax=ax)
    st.pyplot(fig)

    st.subheader('Budget vs Revenue by Cluster')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='budget', y='revenue', hue='cluster', data=movies, ax=ax)
    st.pyplot(fig)

    st.subheader('Correlation Heatmap')
    correlation_matrix = movies[['budget', 'revenue', 'vote_average', 'vote_count']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader('Top 10 Movie Genres')
    genre_counts = movies['genres'].str.split().explode().value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    genre_counts.plot(kind='bar', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader('Number of Movies Released by Year')
    movies['release_year'] = pd.to_datetime(movies['release_date']).dt.year
    year_counts = movies['release_year'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(15, 6))
    year_counts.plot(kind='line', ax=ax)
    st.pyplot(fig)

    st.subheader('Average Vote by Release Year')
    vote_by_year = movies.groupby('release_year')['vote_average'].mean()
    fig, ax = plt.subplots(figsize=(15, 6))
    vote_by_year.plot(kind='line', ax=ax)
    st.pyplot(fig)

    st.subheader('Top 20 Production Companies')
    top_companies = movies['production_companies'].str.split().explode().value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 8))
    top_companies.plot(kind='bar', ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

if __name__ == '__main__':
    st.sidebar.info('This app provides movie recommendations and insights based on the TMDB 5000 Movie Dataset.')