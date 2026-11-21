import os
import numpy as np
import pandas as pd
import faiss
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Folder that contains papers_clean.csv and embeddings_minilm.npy
FOLDER = os.environ.get('PAPER_DATA_DIR', '/content/gdrive/MyDrive/Arxiv/')


def tidy(s):
    s = s.strip().replace(' .', '.')
    if '.' in s and not s.endswith('.'):
        s = s[:s.rfind('.') + 1]
    return s


@st.cache_resource
def load_all():
    df = pd.read_csv(os.path.join(FOLDER, 'papers_clean.csv'))
    df['clean_abstracts'] = df['clean_abstracts'].fillna('').astype(str)
    df['clean_title'] = df['clean_title'].fillna('').astype(str)
    emb = np.load(os.path.join(FOLDER, 'embeddings_minilm.npy')).astype('float32')
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    tfidf = TfidfVectorizer(stop_words='english', max_features=200000)
    D = tfidf.fit_transform(df['clean_abstracts'])
    name = 'sshleifer/distilbart-cnn-12-6'
    tok = AutoTokenizer.from_pretrained(name)
    sm = AutoModelForSeq2SeqLM.from_pretrained(name)
    sm.to('cuda' if torch.cuda.is_available() else 'cpu')
    return df, index, embedder, tfidf, D, tok, sm


st.set_page_config(page_title='Research Paper Finder', layout='wide')
st.title('Research Paper Finder and Summarizer')
st.caption('Search over about 25,000 arXiv AI/ML papers (up to 2020). Summaries can contain mistakes. Always check the original paper.')

df, index, embedder, tfidf, D, tok, sm = load_all()


def search(query, k, method):
    if method.startswith('Semantic'):
        q = embedder.encode([query], normalize_embeddings=True).astype('float32')
        scores, ids = index.search(q, k)
        return list(zip(ids[0], scores[0]))
    sims = cosine_similarity(tfidf.transform([query]), D)[0]
    ids = np.argsort(-sims)[:k]
    return [(i, sims[i]) for i in ids]


def summarize(text):
    enc = tok([text], return_tensors='pt', truncation=True, max_length=1024).to(sm.device)
    with torch.no_grad():
        out = sm.generate(**enc, max_length=60, min_length=10, num_beams=4)
    return tidy(tok.decode(out[0], skip_special_tokens=True))


if 'summaries' not in st.session_state:
    st.session_state['summaries'] = {}

method = st.radio('Search method', ['Semantic (MiniLM + FAISS)', 'Keyword (TF-IDF)'], horizontal=True)
query = st.text_input('Describe the research topic you are looking for')
k = st.slider('Number of results', 1, 10, 5)

if query:
    for i, score in search(query, k, method):
        row = df.iloc[int(i)]
        pid = str(row['arxiv_id'])
        year = 2000 + int(pid[:2])
        st.subheader(row['clean_title'])
        st.caption('Authors: ' + str(row.get('authors', 'n/a')) + ' | Year: ' + str(year) + ' | Similarity: ' + str(round(float(score), 3)))
        with st.expander('Abstract'):
            st.write(row['clean_abstracts'])
        st.markdown('[Open on arXiv](' + str(row['url']) + ')')
        if st.button('Summarize', key='b' + pid):
            st.session_state['summaries'][pid] = summarize(row['clean_abstracts'])
        if pid in st.session_state['summaries']:
            st.info(st.session_state['summaries'][pid])
