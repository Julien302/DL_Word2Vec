import streamlit as st
import numpy as np
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from sklearn.preprocessing import Normalizer

# Configuration de la page
st.set_page_config(
    page_title="Modèle Word2Vec",
    page_icon="🧠",
    layout="wide"
)

# Cache pour éviter de recharger les données à chaque interaction
@st.cache_resource
def load_word2vec_model():
    """Charge le modèle et les données nécessaires"""
    # Chargement du tokenizer
    with open("Data/tokenizer_new.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    # Paramètres du modèle
    vocab_size = 10000
    embedding_dim = 100 
    
    # Reconstruction du modèle
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(vocab_size, activation='softmax'))
    
    # Chargement des poids uniquement
    model.load_weights("Model/word2vec.h5")
    
    # Extraction des vecteurs d'embedding
    vectors = model.layers[0].get_weights()[0]
    
    # Dictionnaires index <-> mot
    word2idx = {word: idx for word, idx in tokenizer.word_index.items() if idx < vocab_size}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return vectors, word2idx, idx2word

# --- Fonctions utilitaires ---
def cosine_similarity(vec1, vec2):
    """Calcule la similarité cosinus entre deux vecteurs"""
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 0
    return np.dot(vec1, vec2) / denom

def find_closest_words(word_index, vectors, idx2word, number_closest=10):
    """Trouve les mots les plus proches d'un mot donné"""
    query_vector = vectors[word_index]
    similarities = []
    
    for index, vector in enumerate(vectors):
        if index == word_index:
            continue
        similarity = cosine_similarity(query_vector, vector)
        similarities.append((similarity, index))
    
    similarities.sort(reverse=True)
    return [(score, idx2word.get(idx, 'Inconnu')) for score, idx in similarities[:number_closest]]

def compute_analogy(word1_idx, word2_idx, word3_idx, vectors, idx2word, number_closest=10):
    """Calcule une analogie : word1 - word2 + word3"""
    query_vector = vectors[word1_idx] - vectors[word2_idx] + vectors[word3_idx]
    query_vector = Normalizer().fit_transform([query_vector])[0]
    
    similarities = []
    for index, vector in enumerate(vectors):
        # Éviter les mots d'entrée dans les résultats
        if index in [word1_idx, word2_idx, word3_idx]:
            continue
        similarity = cosine_similarity(query_vector, vector)
        similarities.append((similarity, index))
    
    similarities.sort(reverse=True)
    return [(score, idx2word.get(idx, 'Inconnu')) for score, idx in similarities[:number_closest]]

def is_word_valid(word, word2idx, vectors):
    """Vérifie si un mot est valide dans le vocabulaire"""
    return word in word2idx and word2idx[word] < len(vectors)

def display_results(results, title):
    """Affiche les résultats de manière formatée"""
    st.write(f"**{title}**")
    for i, (score, word) in enumerate(results, 1):
        st.write(f"{i}. **{word}** — Similarité : {score:.4f}")

# --- Interface utilisateur ---
st.title("🧠 Modèle Word2Vec - Exploration & Analogies")
st.markdown("---")

# Chargement des données
try:
    vectors, word2idx, idx2word = load_word2vec_model()
    st.success("✅ Modèle chargé avec succès !")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

# Section 1: Recherche de mots similaires
st.header("🔍 Recherche de mots similaires")
col1, col2 = st.columns([2, 1])

with col1:
    search_word = st.text_input(
        "Entrez un mot pour voir les mots les plus similaires :",
        placeholder="Exemple: movie, good, acting, story..."
    )

with col2:
    num_results = st.slider("Nombre de résultats", 5, 20, 10)

if search_word:
    if is_word_valid(search_word, word2idx, vectors):
        word_index = word2idx[search_word]
        
        # Affichage du vecteur (optionnel, dans un expander)
        with st.expander("Voir le vecteur d'embedding"):
            st.write(f"Vecteur de **{search_word}** :")
            st.write(vectors[word_index])
        
        # Recherche des mots similaires
        similar_words = find_closest_words(word_index, vectors, idx2word, num_results)
        display_results(similar_words, f"Mots les plus similaires à '{search_word}'")
        
    else:
        st.warning("⚠️ Mot non trouvé dans le vocabulaire.")

# Section 2: Analogies sémantiques
st.markdown("---")
st.header("🧠 Analogies sémantiques")
st.markdown("*Format : Mot1 - Mot2 + Mot3 = ?*")
st.markdown("*Exemple : good - bad + great = excellent*")
st.markdown("*Basé sur votre corpus d'avis de films*")

col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    word1 = st.text_input("Mot 1", value="good", placeholder="Exemple: good")
with col2:
    word2 = st.text_input("Mot 2", value="bad", placeholder="Exemple: bad")
with col3:
    word3 = st.text_input("Mot 3", value="great", placeholder="Exemple: great")
with col4:
    analogy_results = st.slider("Résultats", 5, 15, 10, key="analogy_slider")

if st.button("🔍 Calculer l'analogie", type="primary"):
    words = [word1, word2, word3]
    
    if all(words) and all(is_word_valid(w, word2idx, vectors) for w in words):
        indices = [word2idx[w] for w in words]
        analogy_result = compute_analogy(indices[0], indices[1], indices[2], vectors, idx2word, analogy_results)
        
        st.markdown(f"### Résultat de : **{word1}** - **{word2}** + **{word3}**")
        display_results(analogy_result, "Mots les plus probables")
        
    else:
        invalid_words = [w for w in words if w and not is_word_valid(w, word2idx, vectors)]
        if invalid_words:
            st.error(f"❌ Mot(s) non valide(s) : {', '.join(invalid_words)}")
        else:
            st.error("❌ Veuillez remplir tous les champs.")

# Section d'exploration du vocabulaire
st.markdown("---")
st.header("📚 Exploration du vocabulaire")

# Affichage d'un échantillon du vocabulaire
st.subheader("🔤 Mots les plus fréquents du vocabulaire")
vocab_sample = list(word2idx.keys())[:50] if word2idx else []

if vocab_sample:
    # Affichage sous forme de colonnes
    cols = st.columns(5)
    for i, word in enumerate(vocab_sample):
        cols[i % 5].write(f"• {word}")
else:
    st.write("Vocabulaire non disponible")

# Tableau avec des phrases d'exemple basées sur votre corpus d'avis de films
st.subheader("📝 Exemples de phrases du corpus (avis de films)")
sample_sentences = [
    "this movie is very good and entertaining",
    "the film was boring and poorly made", 
    "great acting and wonderful special effects",
    "i enjoyed watching this film with friends",
    "the plot was predictable but fun to watch",
    "terrible acting and bad cinematography",
    "one of the best movies i have ever seen",
    "waste of time and money avoid this film"
]

sentences_df = {
    "Phrase d'exemple": sample_sentences,
    "Sentiment": ["Positif", "Négatif", "Positif", "Positif", "Neutre", "Négatif", "Positif", "Négatif"],
    "Nombre de mots": [len(sentence.split()) for sentence in sample_sentences]
}

import pandas as pd
df = pd.DataFrame(sentences_df)
st.dataframe(df, use_container_width=True)

# Section d'informations
st.markdown("---")
st.info(f"📊 **Informations sur le modèle :**\n"
        f"- Taille du vocabulaire : {len(word2idx):,} mots\n"
        f"- Dimension des vecteurs : {vectors.shape[1]}\n"
        f"- Algorithme : Similarité cosinus")

# Sidebar avec des exemples basés sur un vocabulaire de critiques de films
with st.sidebar:
    st.header("💡 Exemples d'analogies")
    st.markdown("""
    **Exemples avec votre corpus d'avis de films :**
    - good - bad + terrible = awful
    - movie - film + story = plot
    - great - terrible + wonderful = amazing
    - watch - watched + see = seen
    
    **Mots fréquents à essayer (films/avis) :**
    - movie, film, story, plot, character
    - good, bad, great, terrible, wonderful
    - watch, see, like, love, hate
    - acting, director, scene, music
    - time, people, way, make, think
    
    **Conseils :**
    - Votre modèle est entraîné sur des avis de films
    - Utilisez des mots du vocabulaire affiché
    - Les mots liés au cinéma donnent de meilleurs résultats
    - Testez d'abord la recherche de mots similaires
    """)