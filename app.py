import streamlit as st
import pickle
import pandas as pd
import requests
import io  # Required for loading bytes into Pandas/Pickle

# --- TMDB Configuration ---
# Your valid TMDB API key has been added here.
TMDB_API_KEY = "a34b778de4aa74ff3cad06a6cd75fb11"
API_BASE_URL = "https://api.themoviedb.org/3/movie/"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500/"  # Reverting to w500 poster size


# --- Helper Function to Fetch Poster ---
# This function calls the TMDB API to get the poster path for a given movie ID.
@st.cache_data(ttl=3600)  # Cache the API calls for 1 hour to prevent excessive requests
def fetch_poster(movie_id):
    """Fetches the poster path for a given TMDB movie ID."""

    # LOGGING: Print the ID being used for debugging
    print(f"DEBUG: Attempting to fetch poster for movie_id: {movie_id}")

    try:
        # We ensure the ID is an integer string, as this is the standard TMDB format
        movie_id_str = str(int(movie_id))
        url = f"{API_BASE_URL}{movie_id_str}?api_key={TMDB_API_KEY}&language=en-US"

        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()

        # Check for explicit API error response (e.g., status code 7 for invalid key)
        if 'status_code' in data and data['status_code'] == 7:
            st.error("TMDB API Key Error: Invalid API key (Code 7). Please check the key.")
            return "https://placehold.co/500x750/cccccc/333333?text=Invalid+API+Key"

        poster_path = data.get('poster_path')
        if poster_path:
            # LOGGING: Show the successful URL
            print(f"DEBUG: Successfully retrieved poster URL: {POSTER_BASE_URL + poster_path}")
            return POSTER_BASE_URL + poster_path
        else:
            # Use a placeholder if no poster is found
            print(f"DEBUG: Poster not available for movie_id: {movie_id}")
            return "https://placehold.co/500x750/cccccc/333333?text=Poster+Unavailable"

    except requests.exceptions.HTTPError as e:
        # Catch specific HTTP errors (like 404 Not Found)
        if e.response.status_code == 404:
            st.warning(f"Movie ID {movie_id} not found on TMDB. Skipping poster.")
            # LOGGING: Show 404 error
            print(f"ERROR: 404 Not Found for movie_id: {movie_id}")
            return "https://placehold.co/500x750/cccccc/333333?text=ID+Not+Found"
        else:
            st.error(f"HTTP Error fetching poster: {e}. Check network/API setup.")
            return "https://placehold.co/500x750/cccccc/333333?text=Network+Error"

    except requests.exceptions.RequestException as e:
        # Handle general network errors (DNS failure, connection refused, etc.)
        st.error(f"Connection Error fetching poster: {e}")
        return "https://placehold.co/500x750/cccccc/333333?text=Connection+Refused"

    except Exception as e:
        # Fallback for other errors (e.g., JSON parsing or ValueError from int(movie_id))
        st.error(f"Unknown Error in fetch_poster: {e}")
        # --- FIX: Removed extraneous parenthesis here ---
        return "https://placehold.co/500x750/cccccc/333333?text=Unknown+Error"


# --- Recommendation Function ---
def recommend(movie):
    """Finds 5 similar movies and fetches their posters."""
    try:
        # 1. Find the index of the selected movie
        movie_index = movies[movies['title'] == movie].index[0]

        # 2. Get similarity scores for that movie (from the cosine similarity matrix)
        distances = similarity[movie_index]

        # 3. Sort the distances (similarity scores) and get the top 5 (indices 1 to 6)
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        recommended_movie_posters = []

        # 4. Extract movie titles and fetch posters
        for i in movies_list:
            # Added a type cast just in case the value is not a simple integer
            movie_id = int(movies.iloc[i[0]].movie_id)

            recommended_movies.append(movies.iloc[i[0]].title)
            recommended_movie_posters.append(fetch_poster(movie_id))

        return recommended_movies, recommended_movie_posters

    except IndexError:
        st.error(f"Movie '{movie}' not found in the dataset. Please select another movie.")
        return [], []
    except Exception as e:
        st.error(f"An unexpected error occurred during recommendation: {e}")
        return [], []


# --- Data Loading (Crucial Step: Loading your .pkl files) ---
@st.cache_data
def load_data():
    """Loads the pickled data files."""
    try:
        # Load the movies DataFrame
        with open('movies.pkl', 'rb') as f:
            movies = pickle.load(f)

        # Load the similarity matrix
        with open('similarity.pkl', 'rb') as f:
            similarity = pickle.load(f)

        # CRITICAL CHECK: Ensure 'movie_id' column exists and is numeric
        if 'movie_id' not in movies.columns:
            st.error("Error: The 'movies.pkl' file does not contain a 'movie_id' column!")
            return pd.DataFrame(), None

        # Print a sample of the IDs to the console to check the data format
        print("DEBUG: Sample Movie IDs from DataFrame:")
        print(movies[['title', 'movie_id']].head())

        return movies, similarity

    except FileNotFoundError:
        st.error(
            "Required data files (movies.pkl, similarity.pkl) not found. Please ensure they are in the same directory as this script.")
        return pd.DataFrame(), None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), None


# Load the data globally
movies, similarity = load_data()
movie_list = movies['title'].values if not movies.empty else []

# --- Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="Movie Recommender System")

# Custom CSS for a better look (using a slightly retro movie theme)
st.markdown("""
<style>
    /* Main Streamlit container adjustments */
    .stApp {
        background-color: #1c1c1c; /* Dark background */
        color: #f0f0f0; /* Light text */
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #e50914; /* Netflix Red/Movie theme color */
        text-align: center;
        padding-bottom: 15px;
        border-bottom: 2px solid #333333;
    }
    /* Style for the selectbox and button */
    .stSelectbox, .stButton > button {
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #e50914; 
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #f40a1b;
    }
    /* Style for the movie cards */
    .movie-card {
        background-color: #2a2a2a;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        height: 100%; /* Important for alignment */
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    .movie-title {
        color: #ffffff;
        font-size: 1.1em;
        margin-top: 10px;
        min-height: 40px; /* Ensure titles take up space */
    }
    /* Ensure the posters are responsive */
    .stImage > img {
        border-radius: 6px;
        max-width: 100%;
        height: auto;
    }
</style>
""", unsafe_allow_html=True)

st.title('🎬 Cinematic Matchmaker: Movie Recommendation System')

if not movies.empty and similarity is not None:
    # Dropdown for movie selection
    selected_movie_name = st.selectbox(
        'Choose a movie from the list:',
        movie_list
    )

    # Recommendation button
    if st.button('Show Recommendations'):
        with st.spinner('Finding the best matches...'):
            names, posters = recommend(selected_movie_name)

        if names:
            st.markdown("---")
            st.subheader(f"Top 5 Recommendations based on '{selected_movie_name}':")

            # Use Streamlit columns to display the results side-by-side
            col1, col2, col3, col4, col5 = st.columns(5)

            # Display each recommended movie in a column
            cols = [col1, col2, col3, col4, col5]

            # Iterate over the maximum possible recommendations (5)
            for i in range(len(names)):
                with cols[i]:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    # The image function will load the URL, whether it's a real poster or a placeholder
                    st.image(posters[i], caption=names[i])
                    st.markdown(f'<div class="movie-title">{names[i]}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("Please resolve the data loading issues mentioned above to proceed.")
