
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

# =========================
# CONFIGURATION DE LA PAGE
# =========================
st.set_page_config(
    page_title="Cinéma Creuse - Plateforme Complète",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
/* =========================
   UPGRADE DESIGN CINÉMA
   ========================= */

/* Sections visuelles (cards naturelles) */
section[data-testid="stVerticalBlock"] > div:has(h2),
section[data-testid="stVerticalBlock"] > div:has(h3) {
    background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(0,0,0,0.4));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 30px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
}

/* Séparateurs plus élégants */
hr {
    height: 2px;
    background: linear-gradient(90deg, transparent, #e50914, transparent);
    margin: 3rem 0;
}

/* Titres animés subtils */
h1, h2 {
    position: relative;
}

h1::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: -8px;
    width: 60%;
    height: 4px;
    background: linear-gradient(90deg, #e50914, transparent);
    border-radius: 4px;
}

/* Effet glassmorphism sur les métriques */
div[data-testid="metric-container"] {
    backdrop-filter: blur(8px);
    background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(0,0,0,0.4));
    border: 1px solid rgba(255,255,255,0.1);
}

/* Graphiques : encadrement premium */
.js-plotly-plot {
    background: linear-gradient(145deg, rgba(255,255,255,0.02), rgba(0,0,0,0.4));
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
}

/* Sidebar : plus cinéma */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] label {
    color: #ffffff !important;
}

/* Hover global doux */
section:hover {
    transition: all 0.3s ease;
}

/* Texte principal un peu plus lisible */
p, li {
    font-size: 1.02rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CSS PERSONNALISÉ
# =========================
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */

    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1C1C1C 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141414 0%, #0a0a0a 100%);
        border-right: 1px solid #2a2a2a;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2.5rem;
        letter-spacing: -0.02em;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        border-bottom: 3px solid #e50914;
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    h2 {
        color: #e0e0e0;
        font-weight: 600;
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #b0b0b0;
        font-weight: 500;
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #808080;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #1a1a1a, #0f0f0f);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2a2a2a;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Cards & Containers */
    .element-container {
        background: transparent;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #e50914 0%, #b20710 100%);
        color: #ffffff;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #f40612 0%, #c91017 100%);
        box-shadow: 0 6px 16px rgba(229, 9, 20, 0.6);
        transform: translateY(-2px);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        color: #ffffff;
        border-radius: 8px;
    }
    
    .stSelectbox label {
        color: #b0b0b0;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    
    /* Radio Buttons */
    .stRadio > label {
        color: #b0b0b0;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.85rem;
    }
    
    .stRadio > div {
        background: transparent;
    }
    
    [data-baseweb="radio"] > div {
        background-color: #1a1a1a;
        border: 2px solid #2a2a2a;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #0f0f0f;
        border-radius: 8px;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #808080;
        padding: 0.75rem 2rem;
        font-weight: 500;
        border-radius: 6px;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #e50914 0%, #b20710 100%);
        color: #ffffff;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #2a2a2a 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Images */
    img {
        border-radius: 8px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    img:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.8);
    }
    
    /* Movie Poster Specific */
    [data-testid="stImage"] > img {
        height: 600px !important;
        width: 100% !important;
        object-fit: cover !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        color: #b0b0b0;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #222222;
        border-color: #e50914;
    }
    
    .streamlit-expanderContent {
        background-color: #0f0f0f;
        border: 1px solid #2a2a2a;
        border-top: none;
        color: #b0b0b0;
    }
    
    /* Info/Success/Warning Boxes */
    .stAlert {
        background-color: #1a1a1a;
        border-left: 4px solid #e50914;
        color: #e0e0e0;
        border-radius: 8px;
        padding: 1rem 1.5rem;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: #0f0f0f;
    }
    
    /* Text */
    p, li, span {
        color: #b0b0b0;
        line-height: 1.6;
    }
    
    strong {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Caption */
    .caption {
        color: #707070;
        font-size: 0.85rem;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2a2a2a;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #3a3a3a;
    }
    
    /* Plotly Dark Theme */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 20px;
    }
            
    /* Format unique pour toutes les images */
    img {
        width: 100% !important;
        height: 500px !important;
        object-fit: cover !important;
        border-radius: 14px;
    }
    
    /* Images dans les colonnes (recommandations) */
    div[data-testid="stImage"] > img {
        height: 350px !important;
    }
    </style>
""", unsafe_allow_html=True)


# CLASSES D'EXTRACTION DE DONNÉES

class INSEEDataExtractor:
    def get_population_data(self):
        data = {
            'Tranche_age': ['0-14 ans', '15-29 ans', '30-44 ans', '45-59 ans', '60-74 ans', '75 ans et +'],
            'Population': [13800, 12500, 15200, 22100, 28400, 23500],
            'Pourcentage': [11.9, 10.8, 13.2, 19.1, 24.6, 20.4]
        }
        return pd.DataFrame(data)

class CNCDataExtractor:
    def get_top_films_2024(self):
        data = {
            'Film': ['Un p\'tit truc en plus', 'Le Comte de Monte-Cristo', 'Vice-versa 2', 'Vaiana 2', 
                     "L'Amour Ouf", 'Moi, moche et méchant 4', 'Dune 2', 'Deadpool & Wolverine', 
                     'Gladiator 2', 'Mufasa: Le Roi Lion'],
            'Genre': ['Comédie','Aventure', 'Animation', 'Animation', 'Drame', 'Animation', 
                      'Science-Fiction', 'Action', 'Action', 'Animation'],
            'Entrées_millions': [10.72, 9.39, 8.29, 6.68, 4.83, 4.38, 4.14, 3.63, 2.91, 2.54],
            'Type': ['Français', 'Français', 'US', 'US', 'Français', 'US', 'US', 'Autres', 'Autres', 'US']
        }
        return pd.DataFrame(data)

    def get_frequentation_nationale(self):
        data = {
            'Année': [2019, 2020, 2021, 2022, 2023, 2024],
            'Entrées_millions': [213.3, 65.1, 95.5, 152.1, 180.4, 181.5],
            'Nombre_films': [746, 364, 454, 676, 716, 744]
        }
        df = pd.DataFrame(data)
        df['Evolution_%'] = df['Entrées_millions'].pct_change() * 100
        return df

    def get_frequentation_creuse(self):
        data = {
          'Catégorie': ['Art et Essai', 'Art et Essai', 'Films Français', 'Films Français', 
                        'Films Américains', 'Films Américains'],
          'Entité': ['Creuse (23)', 'Moyenne Nationale', 'Creuse (23)', 'Moyenne Nationale', 
                     'Creuse (23)', 'Moyenne Nationale'],
          'Part de marché (%)': [30.5, 25.1, 56.0, 44.8, 27.8, 36.3]
        }
        return pd.DataFrame(data)


# FONCTIONS DE CHARGEMENT DES DONNÉES

@st.cache_data
def load_movie_data():
    """Charge les données pour le système de recommandation"""
    df = pd.read_csv('https://raw.githubusercontent.com/Yachre/Creuze/refs/heads/main/Database_finale.csv')
    columns_to_combine = ['Genre', 'Réalisateur', 'Acteur', 'Actrice', 'Synopsis']
    for col in columns_to_combine:
        df[col] = df[col].fillna('')
    df['features'] = df['Genre'] + " " + df['Réalisateur'] + " " + \
                     df['Acteur'] + " " + df['Actrice'] + " " + df['Synopsis']
    return df

@st.cache_data
def load_market_data():
    """Charge toutes les données pour l'étude de marché"""
    df_population = pd.DataFrame({
        'Tranche_age': ['0-14 ans', '15-29 ans', '30-44 ans', '45-59 ans', '60-74 ans', '75 ans et +'],
        'Population': [13800, 12500, 15200, 22100, 28400, 23500],
        'Pourcentage': [11.9, 10.8, 13.2, 19.1, 24.6, 20.4]
    })
    
    df_revenus = pd.DataFrame({
        'Indicateur': ['Revenu médian', 'Revenu moyen', 'Taux de pauvreté', 'Écart interdécile'],
        'Creuse': [18166, 19800, 15.2, 3.2],
        'Nouvelle-Aquitaine': [20590, 22100, 13.8, 3.5],
        'France': [22040, 23710, 14.5, 3.6],
        'Unité': ['€/an', '€/an', '%', 'ratio']
    })
    
    df_csp = pd.DataFrame({
        'CSP': ['Agriculteurs', 'Artisans, commerçants', 'Cadres', 'Professions intermédiaires',
                'Employés', 'Ouvriers', 'Retraités'],
        'Creuse_%': [3.5, 5.8, 4.2, 11.3, 14.5, 21.0, 39.7],
        'France_%': [1.2, 6.4, 10.1, 14.2, 16.8, 20.5, 26.8]
    })
    
    df_internet = pd.DataFrame({
        'Type_zone': ['Urbain', 'Rural dense', 'Rural intermédiaire', 'Rural isolé'],
        'Taux_acces_%': [92, 87, 82, 75],
        'Débit_moyen_Mbps': [85, 45, 30, 15]
    })
    
    df_freq_creuse = pd.DataFrame({
        'Ville': ['Guéret', 'Aubusson', 'La Souterraine', 'Bourganeuf', 'Évaux-les-Bains'],
        'Cinema': ['Le Sénéchal', 'Le Colbert', "L'Eden", 'Claude Miller', 'Cinéma municipal'],
        'Nb_salles': [5, 2, 1, 1, 1],
        'Entrées_2023': [92792, 19800, 11936, 10500, 13500],
        'Entrées_2024': [94792, 21600, 12305, 11700, 13747],
        'Evolution_%': [2.1, 9.0, 3.1, 11.4, 1.8]
    })
    
    df_genres = pd.DataFrame({
        'Genre': ['Action/Aventure', 'Comédie', 'Comédie dramatique', 'Animation',
                  'Thriller/Policier', 'Science-Fiction', 'Drame', 'Horreur', 'Romantique', 'Comédie musicale'],
        'Preference_jeunes_%': [82, 73, 55, 60, 48, 45, 35, 30, 28, 20],
        'Preference_seniors_%': [45, 78, 68, 40, 35, 25, 65, 15, 55, 35],
        'Preference_familles_%': [70, 85, 60, 90, 25, 55, 40, 10, 45, 50]
    })
    df_genres['Moyenne_%'] = df_genres[['Preference_jeunes_%', 'Preference_seniors_%', 'Preference_familles_%']].mean(axis=1)
    
    df_top_films = pd.DataFrame({
        'Film': ["Un p'tit truc en plus", 'Le Comte de Monte-Cristo', 'Vice-versa 2', 'Vaiana 2',
                 "L'Amour Ouf", 'Moi, moche et méchant 4', 'Dune 2', 'Deadpool & Wolverine', 
                 'Gladiator 2', 'Mufasa: Le Roi Lion'],
        'Genre': ['Comédie', 'Aventure', 'Animation', 'Animation', 'Drame', 'Animation',
                  'Science-Fiction', 'Action', 'Action', 'Animation'],
        'Entrées_millions': [10.72, 9.39, 8.29, 6.68, 4.83, 4.38, 4.14, 3.63, 2.91, 2.54],
        'Type': ['Français', 'Français', 'US', 'US', 'Français', 'US', 'US', 'UK', 'UK', 'US']
    })
    
    df_freq_nat = pd.DataFrame({
        'Année': [2019, 2020, 2021, 2022, 2023, 2024],
        'Entrées_millions': [213.3, 65.1, 95.5, 152.1, 180.4, 181.5],
        'Nombre_films': [746, 364, 454, 676, 716, 744]
    })
    
    df_saison = pd.DataFrame({
        'Mois': ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
                'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre'],
        'Indice_freq': [85, 90, 75, 70, 80, 90, 120, 110, 85, 95, 100, 130],
        'Vacances_scolaires': ['Non', 'Oui', 'Non', 'Oui', 'Non', 'Non',
                              'Oui', 'Oui', 'Non', 'Oui', 'Non', 'Oui']
    })
    
    return (df_population, df_revenus, df_csp, df_internet, df_freq_creuse, 
            df_genres, df_top_films, df_freq_nat, df_saison)

@st.cache_data(show_spinner=False)
def traduire_en_francais(texte):
    """Traduit le texte en français"""
    if not isinstance(texte, str) or texte.strip() == "":
        return ""
    try:
        return GoogleTranslator(source="auto", target="fr").translate(texte)
    except Exception:
        return texte

def get_recommendations(title, df, sig):
    """Obtient les recommandations de films similaires"""
    idx = df.index[df['Titre'] == title].tolist()[0]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sig_scores[1:7]]
    return df.iloc[movie_indices]

# =========================
# SIDEBAR - MENU DE NAVIGATION
# =========================

st.sidebar.title("Navigation")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Choisissez une section :",
    ["Accueil", "Étude de Marché", "KPI Stratégiques", "Recommandation de Films"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**À propos**  
Plateforme d'analyse et de recommandation pour le cinéma en Creuse.

**Sources :** INSEE, CNC  
**Année :** 2024
""")

# =========================
# PAGE D'ACCUEIL
# =========================

if menu == "Accueil":
    st.title("Plateforme Cinéma Creuse")
    st.markdown("### Votre outil complet d'analyse et de recommandation")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Étude de Marché
        
        Analyse complète du marché du cinéma dans la Creuse :
        - Démographie INSEE
        - Fréquentation des salles
        - Préférences de genres
        - Analyses approfondies
        """)
        
    with col2:
        st.markdown("""
        ### KPI Stratégiques
        
        Indicateurs clés de performance :
        - Comparaison Creuse vs National
        - Top 10 Box-Office 2024
        - Profil démographique
        - Tendances du marché
        """)
        
    with col3:
        st.markdown("""
        ### Recommandation
        
        Système intelligent de recommandation :
        - Base de données complète
        - Algorithme de similarité
        - Suggestions personnalisées
        - Synopsis traduits
        """)
    
    st.markdown("---")
    
    st.success("""
       **Comment utiliser cette plateforme ?**
    
    1. **Étude de Marché** : Consultez les analyses démographiques et de fréquentation
    2. **KPI Stratégiques** : Visualisez les indicateurs clés pour la prise de décision
    3. **Recommandation de Films** : Trouvez des films similaires selon vos préférences
    
    Utilisez le menu latéral pour naviguer entre les sections.
    """)

# =========================
# PAGE ÉTUDE DE MARCHÉ
# =========================

elif menu == "Étude de Marché":
    st.title("Étude de Marché : Cinéma en Creuse")
    st.markdown("---")
    
    # Chargement des données
    (df_population, df_revenus, df_csp, df_internet, df_freq_creuse, 
     df_genres, df_top_films, df_freq_nat, df_saison) = load_market_data()
    
    # Sous-menu
    section = st.radio(
        "Navigation :",
        ["Vue d'ensemble", "Démographie INSEE", "Fréquentation Cinémas",
         "Préférences & Tendances", "Analyses Approfondies"],
        horizontal=True
    )
    
    # VUE D'ENSEMBLE
    if section == "Vue d'ensemble":
        st.header("Tableau de Bord - Vue d'Ensemble")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Population Creuse", f"{df_population['Population'].sum():,}", "habitants")
        with col2:
            st.metric("Entrées 2024", f"{df_freq_creuse['Entrées_2024'].sum():,}", 
                     f"+{df_freq_creuse['Evolution_%'].mean():.1f}%")
        with col3:
            st.metric("Cinémas actifs", len(df_freq_creuse), f"{df_freq_creuse['Nb_salles'].sum()} salles")
        with col4:
            st.metric("Revenu médian", 
                     f"{df_revenus[df_revenus['Indicateur']=='Revenu médian']['Creuse'].values[0]:,.0f} €",
                     "-17.6% vs France")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fréquentation par Cinéma (2024)")
            fig = px.bar(df_freq_creuse.sort_values('Entrées_2024', ascending=True),
                        x='Entrées_2024', y='Cinema', orientation='h',
                        color='Entrées_2024', color_continuous_scale='Teal', text='Entrées_2024')
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("👥 Structure de la Population")
            fig = px.pie(df_population, values='Population', names='Tranche_age', hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📈 Évolution de la Fréquentation Nationale")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_freq_nat['Année'], y=df_freq_nat['Entrées_millions'],
                                mode='lines+markers', name='Entrées (millions)',
                                line=dict(color='#1f77b4', width=3), marker=dict(size=10)))
        fig.update_layout(xaxis_title="Année", yaxis_title="Entrées (millions)",
                         height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    # DÉMOGRAPHIE INSEE
    elif section == "Démographie INSEE":
        st.header("Analyse Démographique - Données INSEE")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Population", "Revenus", "CSP", "Internet"])
        
        with tab1:
            st.subheader("Structure par Âge de la Population - Creuse 2023")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(df_population, x='Pourcentage', y='Tranche_age', orientation='h',
                            text='Pourcentage', color='Pourcentage', color_continuous_scale='Blues')
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                fig.update_layout(xaxis_title="Pourcentage de la population (%)",
                                 yaxis_title="Tranche d'âge", showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(df_population[['Tranche_age', 'Population', 'Pourcentage']],
                           hide_index=True, use_container_width=True)
                st.info(f"""
                **Points clés :**
                - Population totale : **{df_population['Population'].sum():,}** habitants
                - Tranche dominante : **60-74 ans** (24.6%)
                - Population vieillissante : **44.7%** ont 60 ans ou plus
                """)
        
        with tab2:
            st.subheader("Comparaison des Revenus")
            df_rev_comp = df_revenus[df_revenus['Indicateur'].isin(['Revenu médian', 'Revenu moyen'])]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Creuse', x=df_rev_comp['Indicateur'], 
                                y=df_rev_comp['Creuse'], marker_color='coral'))
            fig.add_trace(go.Bar(name='Nouvelle-Aquitaine', x=df_rev_comp['Indicateur'],
                                y=df_rev_comp['Nouvelle-Aquitaine'], marker_color='steelblue'))
            fig.add_trace(go.Bar(name='France', x=df_rev_comp['Indicateur'],
                                y=df_rev_comp['France'], marker_color='seagreen'))
            fig.update_layout(barmode='group', yaxis_title="Revenu (€/an)", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Taux de Pauvreté Comparé")
            df_pov = df_revenus[df_revenus['Indicateur'] == 'Taux de pauvreté']
            fig = go.Figure(go.Bar(
                x=['Creuse', 'Nouvelle-Aquitaine', 'France'],
                y=[df_pov['Creuse'].values[0], df_pov['Nouvelle-Aquitaine'].values[0], 
                   df_pov['France'].values[0]],
                marker_color=['coral', 'steelblue', 'seagreen'],
                text=[f"{v}%" for v in [df_pov['Creuse'].values[0], 
                      df_pov['Nouvelle-Aquitaine'].values[0], df_pov['France'].values[0]]],
                textposition='outside'))
            fig.update_layout(yaxis_title="Taux de pauvreté (%)", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Répartition des Catégories Socioprofessionnelles")
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Creuse', x=df_csp['CSP'], y=df_csp['Creuse_%'], 
                                marker_color='coral'))
            fig.add_trace(go.Bar(name='France', x=df_csp['CSP'], y=df_csp['France_%'],
                                marker_color='steelblue'))
            fig.update_layout(barmode='group', xaxis_tickangle=-45, 
                             yaxis_title="Pourcentage (%)", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.warning("""
            **Observations :**
            - Forte surreprésentation des **retraités** (39.7% vs 26.8% France)
            - Sous-représentation des **cadres** (4.2% vs 10.1% France)
            - Population agricole importante (3.5% vs 1.2% France)
            """)
        
        with tab4:
            st.subheader("Accès à Internet en Milieu Rural")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df_internet, x='Type_zone', y='Taux_acces_%',
                            color='Taux_acces_%', color_continuous_scale='Greens',
                            text='Taux_acces_%')
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                fig.update_layout(xaxis_title="Type de Zone", yaxis_title="Taux d'accès (%)",
                                 showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df_internet, x='Type_zone', y='Débit_moyen_Mbps',
                            color='Débit_moyen_Mbps', color_continuous_scale='Purples',
                            text='Débit_moyen_Mbps')
                fig.update_traces(texttemplate='%{text} Mbps', textposition='outside')
                fig.update_layout(xaxis_title="Type de Zone", yaxis_title="Débit Moyen (Mbps)",
                                 showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # FRÉQUENTATION CINÉMAS
    elif section == "Fréquentation Cinémas":
        st.header("Analyse de la Fréquentation des Cinémas")
        
        tab1, tab2, tab3 = st.tabs(["Creuse", "National", "Saisonnalité"])
        
        with tab1:
            st.subheader("Cinémas de la Creuse - Données 2024")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(df_freq_creuse.sort_values('Entrées_2024', ascending=True),
                            x='Entrées_2024', y='Cinema', orientation='h', color='Evolution_%',
                            color_continuous_scale='RdYlGn', text='Entrées_2024',
                            hover_data=['Ville', 'Nb_salles', 'Evolution_%'])
                fig.update_traces(texttemplate='%{text:,}', textposition='outside')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(df_freq_creuse[['Cinema', 'Ville', 'Nb_salles', 'Entrées_2024', 'Evolution_%']],
                           hide_index=True, use_container_width=True)
            
            st.subheader("Évolution 2023 → 2024")
            fig = go.Figure()
            fig.add_trace(go.Bar(name='2023', x=df_freq_creuse['Cinema'],
                                y=df_freq_creuse['Entrées_2023'], marker_color='lightcoral'))
            fig.add_trace(go.Bar(name='2024', x=df_freq_creuse['Cinema'],
                                y=df_freq_creuse['Entrées_2024'], marker_color='darkturquoise'))
            fig.update_layout(barmode='group', xaxis_tickangle=-45,
                             yaxis_title="Nombre d'entrées", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total entrées 2024", f"{df_freq_creuse['Entrées_2024'].sum():,}")
            with col2:
                st.metric("Évolution moyenne", f"+{df_freq_creuse['Evolution_%'].mean():.1f}%")
            with col3:
                st.metric("Total salles", df_freq_creuse['Nb_salles'].sum())
        
        with tab2:
            st.subheader("Évolution de la Fréquentation Nationale")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(name='Entrées (millions)', x=df_freq_nat['Année'],
                                y=df_freq_nat['Entrées_millions'], marker_color='steelblue'),
                         secondary_y=False)
            fig.add_trace(go.Scatter(name='Nombre de films', x=df_freq_nat['Année'],
                                    y=df_freq_nat['Nombre_films'], mode='lines+markers',
                                    marker_color='coral', line=dict(width=3)), secondary_y=True)
            fig.update_xaxes(title_text="Année")
            fig.update_yaxes(title_text="Entrées (millions)", secondary_y=False)
            fig.update_yaxes(title_text="Nombre de films", secondary_y=True)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Tendances :**
            - Impact COVID-19 visible en 2020-2021
            - Reprise progressive depuis 2022
            - Fréquentation 2024 : **181.5 millions** d'entrées (+0.6% vs 2023)
            """)
        
        with tab3:
            st.subheader("Saisonnalité de la Fréquentation")
            colors = ['#ff6b6b' if v == 'Oui' else '#4ecdc4' for v in df_saison['Vacances_scolaires']]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_saison['Mois'], y=df_saison['Indice_freq'],
                                marker_color=colors, text=df_saison['Indice_freq'],
                                textposition='outside',
                                hovertemplate='<b>%{x}</b><br>Indice: %{y}<br>Vacances: %{customdata}<extra></extra>',
                                customdata=df_saison['Vacances_scolaires']))
            fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Moyenne (100)")
            fig.update_layout(xaxis_tickangle=-45, yaxis_title="Indice de fréquentation",
                             height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🔴 **Vacances scolaires**")
            with col2:
                st.markdown("🔵 **Hors vacances**")
            
            st.success("""
            **Pics de fréquentation :**
            - **Décembre** : 130 (période des fêtes)
            - **Juillet** : 120 (grandes vacances)
            - **Août** : 110 (grandes vacances)
            """)
    
    # PRÉFÉRENCES & TENDANCES
    elif section == "Préférences & Tendances":
        st.header("Préférences de Genres & Tendances du Marché")
        
        tab1, tab2 = st.tabs(["Genres", "Top Films 2024"])
        
        with tab1:
            st.subheader("Préférences de Genres par Segment de Public")
            df_genres_sorted = df_genres.sort_values('Moyenne_%', ascending=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Jeunes (18-30)', y=df_genres_sorted['Genre'],
                                x=df_genres_sorted['Preference_jeunes_%'], orientation='h',
                                marker_color='coral'))
            fig.add_trace(go.Bar(name='Seniors (50+)', y=df_genres_sorted['Genre'],
                                x=df_genres_sorted['Preference_seniors_%'], orientation='h',
                                marker_color='steelblue'))
            fig.add_trace(go.Bar(name='Familles', y=df_genres_sorted['Genre'],
                                x=df_genres_sorted['Preference_familles_%'], orientation='h',
                                marker_color='seagreen'))
            fig.add_trace(go.Bar(name='Moyenne', y=df_genres_sorted['Genre'],
                                x=df_genres_sorted['Moyenne_%'], orientation='h',
                                marker_color='gold'))
            fig.update_layout(barmode='group', xaxis_title="Pourcentage de préférence (%)",
                             height=600, legend=dict(orientation="h", yanchor="bottom", 
                                                    y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Analyse Détaillée par Genre")
            df_display = df_genres[['Genre', 'Preference_jeunes_%', 'Preference_seniors_%',
                                   'Preference_familles_%', 'Moyenne_%']].copy()
            df_display.columns = ['Genre', 'Jeunes', 'Seniors', 'Familles', 'Moyenne']
            df_display = df_display.sort_values('Moyenne', ascending=False)
            st.dataframe(df_display.style.background_gradient(cmap='RdYlGn', subset=['Moyenne']),
                        hide_index=True, use_container_width=True)
            
            st.warning("""
            **Insights pour la Creuse :**
            - **Comédies** : très prisées par tous les segments (moyenne 78%)
            - **Animations** : excellent pour les familles (90%)
            - **Drames et comédies dramatiques** : forte appétence des seniors
            - Adapter la programmation selon la démographie locale (44.7% de 60+)
            """)
        
        with tab2:
            st.subheader("Top 10 Films 2024 en France")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(df_top_films, x='Entrées_millions', y='Film', orientation='h',
                            color='Type',
                            color_discrete_map={'Français': 'steelblue', 'US': 'coral', 'UK': 'seagreen'},
                            text='Entrées_millions', hover_data=['Genre'])
                fig.update_traces(texttemplate='%{text:.2f}M', textposition='outside')
                fig.update_layout(xaxis_title="Entrées (millions)",
                                 yaxis={'categoryorder':'total ascending'}, height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(df_top_films[['Film', 'Genre', 'Entrées_millions', 'Type']],
                           hide_index=True, use_container_width=True)
            
            st.subheader("Répartition par Origine")
            col1, col2 = st.columns(2)
            
            with col1:
                type_sum = df_top_films.groupby('Type')['Entrées_millions'].sum().reset_index()
                fig = px.pie(type_sum, values='Entrées_millions', names='Type', hole=0.4,
                            color_discrete_map={'Français': 'steelblue', 'US': 'coral', 'UK': 'seagreen'})
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                genre_sum = df_top_films.groupby('Genre')['Entrées_millions'].sum().reset_index()
                genre_sum = genre_sum.sort_values('Entrées_millions', ascending=False)
                fig = px.bar(genre_sum, x='Genre', y='Entrées_millions',
                            color='Entrées_millions', color_continuous_scale='Viridis')
                fig.update_layout(xaxis_tickangle=-45, showlegend=False,
                                 yaxis_title="Entrées (millions)")
                st.plotly_chart(fig, use_container_width=True)
    
    # ANALYSES APPROFONDIES
    elif section == "Analyses Approfondies":
        st.header("Analyses Approfondies & Recommandations")
        
        tab1, tab2 = st.tabs(["Synthèse", "Recommandations"])
        
        with tab1:
            st.subheader("Synthèse de l'Étude de Marché")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Profil Démographique")
                st.info(f"""
                - Population : **{df_population['Population'].sum():,}** habitants
                - 60+ ans : **44.7%** (vieillissement marqué)
                - Revenu médian : **18,166 €/an** (-17.6% vs France)
                - Taux de pauvreté : **15.2%**
                - Retraités : **39.7%** de la population active
                """)
                
                st.markdown("### Marché Cinéma")
                st.success(f"""
                - **{len(df_freq_creuse)}** cinémas actifs
                - **{df_freq_creuse['Nb_salles'].sum()}** salles au total
                - **{df_freq_creuse['Entrées_2024'].sum():,}** entrées en 2024
                - Évolution : **+{df_freq_creuse['Evolution_%'].mean():.1f}%**
                - Leader : **Le Sénéchal** (Guéret) - 94,792 entrées
                """)
            
            with col2:
                st.markdown("### Préférences Culturelles")
                top_3_genres = df_genres.nlargest(3, 'Moyenne_%')
                st.warning(f"""
                **Top 3 Genres (moyenne) :**
                1. **{top_3_genres.iloc[0]['Genre']}** : {top_3_genres.iloc[0]['Moyenne_%']:.1f}%
                2. **{top_3_genres.iloc[1]['Genre']}** : {top_3_genres.iloc[1]['Moyenne_%']:.1f}%
                3. **{top_3_genres.iloc[2]['Genre']}** : {top_3_genres.iloc[2]['Moyenne_%']:.1f}%
                
                **Films français** : forte performance (3 dans le top 10)
                """)
                
                st.markdown("### Saisonnalité")
                best_month = df_saison.loc[df_saison['Indice_freq'].idxmax()]
                worst_month = df_saison.loc[df_saison['Indice_freq'].idxmin()]
                st.info(f"""
                - Meilleur mois : **{best_month['Mois']}** (indice {best_month['Indice_freq']})
                - Moins bon : **{worst_month['Mois']}** (indice {worst_month['Indice_freq']})
                - Impact vacances scolaires : **+25% en moyenne**
                """)
        
        with tab2:
            st.subheader("Recommandations Stratégiques")
            st.markdown("### Positionnement & Programmation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                #### Opportunités
                
                **1. Cibler la population senior (44.7% de 60+)**
                - Séances matinales
                - Comédies dramatiques et drames
                - Tarifs préférentiels
                - Confort des salles
                
                **2. Capitaliser sur les vacances scolaires**
                - Programmation familiale renforcée
                - Animations pour enfants
                - Événements spéciaux
                
                **3. Valoriser le cinéma français**
                - Succès avéré (top 10)
                - Appétence locale
                - Partenariats régionaux
                """)
            
            with col2:
                st.markdown("""
                #### Défis à Relever
                
                **1. Contraintes économiques**
                - Revenu médian inférieur (-17.6%)
                - Politique tarifaire adaptée
                - Offres groupées/abonnements
                
                **2. Accessibilité numérique**
                - 25% en zone rurale isolée
                - Débit limité (15 Mbps)
                - Billetterie mobile simplifiée
                - Communication multi-canal
                
                **3. Concurrence du streaming**
                - Valoriser l'expérience cinéma
                - Événements exclusifs
                - Qualité audio/vidéo supérieure
                """)
            
            st.markdown("### Programmation Recommandée")
            st.success("""
            **Mix idéal adapté à la Creuse :**
            
            - **35%** : Comédies (tous publics, forte demande)
            - **25%** : Comédies dramatiques & Drames (seniors)
            - **20%** : Animations (familles, vacances)
            - **15%** : Cinéma français d'auteur
            - **5%** : Blockbusters internationaux
            
            **Stratégie saisonnière :**
            - Janvier-Mars : Drames, films d'auteur
            - Avril-Juin : Comédies, avant-premières
            - Juillet-Août : Animations, films familiaux
            - Septembre-Novembre : Rentrée culturelle
            - Décembre : Blockbusters, films de Noël
            """)

# =========================
# PAGE KPI STRATÉGIQUES
# =========================

elif menu == "KPI Stratégiques":
    st.title("Dashboard Stratégique : Cinéma en Creuse (23)")
    
    # Initialisation
    cnc = CNCDataExtractor()
    insee = INSEEDataExtractor()
    
    df_top = cnc.get_top_films_2024()
    df_nat = cnc.get_frequentation_nationale()
    df_creuse = cnc.get_frequentation_creuse()
    df_pop = insee.get_population_data()
    
    # Ligne 1 : Les chiffres clés du CNC
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Part Films FR (23)", "56%", "+11.2% vs Nat")
    with col2:
        st.metric("Part Art & Essai (23)", "30.5%", "+5.4% vs Nat")
    with col3:
        st.metric("Entrées Nat. 2024", "181.5M", "+0.6%")
    
    st.divider()
    
    # Ligne 2 : Démographie
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Profil Démographique (Creuse)")
        fig_pop = px.pie(df_pop, values='Pourcentage', names='Tranche_age', hole=0.4,
                         title="Répartition de la population par âge",
                         color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig_pop, use_container_width=True)
    
    with col_b:
        st.subheader("Comparatif Marché")
        fig_bar = px.bar(df_creuse, x='Catégorie', y='Part de marché (%)', color='Entité',
                         barmode='group', text_auto=True, title="Creuse vs Moyenne Nationale")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Ligne 3 : Top 10
    st.subheader("Top 10 Box-Office 2024")
    fig_h = px.bar(df_top, x='Entrées_millions', y='Film', color='Type', orientation='h',
                   title="Succès Nationaux (Potentiel Creuse)",
                   color_discrete_map={'Français': '#003366', 'US': '#6699CC', 'Autres': "#999999"})
    st.plotly_chart(fig_h, use_container_width=True)
    
    st.success("""
    ** Note pour l'algorithme de recommandation :**
    Le public creusois est majoritairement âgé de plus de 45 ans (64.1% de la population). 
    Cela explique la corrélation entre les KPIs du CNC (succès des films Français et Art & Essai) 
    et la démographie INSEE. Priorisez les genres 'Comédie Dramatique' et 'Drame' dans vos suggestions.
    """)

# =========================
# PAGE RECOMMANDATION DE FILMS
# =========================

elif menu == "Recommandation de Films":
    st.title("Movie Finder & Recommender")
    
    # Chargement des données
    df = load_movie_data()
    base_url = "https://image.tmdb.org/t/p/w500"
    
    # Préparation du moteur TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Barre de sélection
    selected_movie_name = st.selectbox(
        "Recherchez ou sélectionnez un film :",
        df['Titre'].values
    )
    
    # SECTION 1 : DÉTAILS DU FILM SÉLECTIONNÉ
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
            st.subheader(f"Année : {int(movie_info['Année_de_Sortie']) if pd.notnull(movie_info['Année_de_Sortie']) else 'N/A'}")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Note", f" {movie_info['Note']}/10")
            m2.metric("Durée", f" {movie_info['Durée']} min")
            
            st.write(f"**Genre :** {movie_info['Genre']}")
            st.write(f"**Réalisateur :** {movie_info['Réalisateur']}")
            st.write(f"**Casting :** {movie_info['Acteur']}, {movie_info['Actrice']}")
            st.write("**Synopsis :**")
            st.write(traduire_en_francais(movie_info['Synopsis']))
    
    # SECTION 2 : RECOMMANDATIONS
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
                st.caption(f"Note: {row['Note']} | {int(row['Année_de_Sortie']) if pd.notnull(row['Année_de_Sortie']) else ''}")
                
                with st.expander("Lire le synopsis"):
                    st.write(traduire_en_francais(row['Synopsis']))

# =========================
# FOOTER
# =========================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Plateforme Cinéma Creuse 2024</p>
    <p>Sources : INSEE, CNC, TMDB | Développé avec Streamlit</p>
</div>
""", unsafe_allow_html=True)