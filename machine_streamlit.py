import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

# 1. Configuration de la page
st.set_page_config(page_title="Movie Recommender", layout="wide")

# FORMAT UNIQUE POUR TOUTES LES IMAGES

st.markdown(
    """
    <style>
    /* Toutes les images Streamlit */
    img {
        width: 100% !important;
        height: 800px !important;      /* hauteur fixe */
        object-fit: cover !important;  /* recadrage propre */
        border-radius: 14px;
    }

    /* Images dans les colonnes (recommandations) */
    div[data-testid="stImage"] > img {
        height: 700px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data 
def load_data():
    df = pd.read_csv(r'C:\Users\kinga\streamlit quete\data\Database_finale.csv')
    # Nettoyage des colonnes textuelles
    columns_to_combine = ['Genre', 'Réalisateur', 'Acteur', 'Actrice', 'Synopsis']
    for col in columns_to_combine:
        df[col] = df[col].fillna('')
    
    # Création de la soupe de mots pour le moteur de recommandation
    df['features'] = df['Genre'] + " " + df['Réalisateur'] + " " + \
                     df['Acteur'] + " " + df['Actrice'] + " " + df['Synopsis']
    return df

@st.cache_data(show_spinner=False)
def traduire_en_francais(texte):
    if not isinstance(texte, str) or texte.strip() == "":
        return ""
    try:
        return GoogleTranslator(source="auto", target="fr").translate(texte)
    except Exception:
        return texte

def get_recommendations(title, df, sig):
    idx = df.index[df['Titre'] == title].tolist()[0]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    # On prend les indices 1 à 7 (le 0 étant le film lui-même)
    movie_indices = [i[0] for i in sig_scores[1:7]]
    return df.iloc[movie_indices]

# --- CHARGEMENT ---
df = load_data()
base_url = "https://image.tmdb.org/t/p/w500"

# Préparation du moteur TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- INTERFACE STREAMLIT ---
st.title("🎬 Movie Finder & Recommender")

# Barre de sélection
selected_movie_name = st.selectbox(
    "Recherchez ou sélectionnez un film :",
    df['Titre'].values
)

# --- SECTION 1 : DÉTAILS DU FILM SÉLECTIONNÉ ---
if selected_movie_name:
    movie_info = df[df['Titre'] == selected_movie_name].iloc[0]
    
    st.markdown("---")
    col_img, col_det = st.columns([1, 2])
    
    with col_img:
        path = movie_info['Affiche_de_Film']
        img_url = base_url + str(path) if pd.notnull(path) else "https://via.placeholder.com/500x750?text=No+Image"
        st.image(img_url, use_container_width=True)
        
    with col_det:
        st.header(movie_info['Titre'])
        st.subheader(f"📅 Année : {int(movie_info['Année_de_Sortie']) if pd.notnull(movie_info['Année_de_Sortie']) else 'N/A'}")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Note", f"⭐ {movie_info['Note']}/10")
        m2.metric("Durée", f"⏱️ {movie_info['Durée']} min")
        
        st.write(f"**Genre :** {movie_info['Genre']}")
        st.write(f"**Réalisateur :** {movie_info['Réalisateur']}")
        st.write(f"**Casting :** {movie_info['Acteur']}, {movie_info['Actrice']}")
        st.write("**Synopsis :**")
        st.write(traduire_en_francais(movie_info['Synopsis']))

# --- SECTION 2 : RECOMMANDATIONS ---
st.markdown("---")
if st.button('Obtenir des recommandations similaires'):
    recommendations = get_recommendations(selected_movie_name, df, cosine_sim)
    
    st.subheader("Les utilisateurs ont aussi aimé :")
    
    rec_cols = st.columns(3)
    for i, (index, row) in enumerate(recommendations.iterrows()):
        with rec_cols[i % 3]:
            placeholder_url = "https://image.noelshack.com/fichiers/2026/05/3/1769612385-adobe-express-file.png"

            r_path = row['Affiche_de_Film']
            path_str = str(r_path).strip().lower()

            if pd.notnull(r_path) and path_str != "" and "unknow" not in path_str:
                r_img_url = base_url + str(r_path)
            else:
                r_img_url = placeholder_url 
                
            st.image(r_img_url, use_container_width=True)
            st.write(f"**{row['Titre']}**")
            st.caption(f"⭐ Note: {row['Note']} | {int(row['Année_de_Sortie']) if pd.notnull(row['Année_de_Sortie']) else ''}")
            
            with st.expander("Lire le synopsis"):
                st.write(traduire_en_francais(row['Synopsis']))
