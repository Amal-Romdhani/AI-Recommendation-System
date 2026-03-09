"""
Siratify — Système de Recommandation de Posts
Application de démonstration Streamlit

Usage :
    streamlit run app.py

Dépendances : streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import math, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Siratify — Recommandation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800; color: #1B4F8A;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem; color: #2980B9; text-align: center; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F0F6FC; border-left: 4px solid #1B4F8A;
        padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 0.5rem;
    }
    .rec-card {
        background: white; border: 1px solid #E0E0E0; border-radius: 8px;
        padding: 0.8rem 1rem; margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .tag-pill {
        display: inline-block; background: #D6E4F0; color: #1B4F8A;
        border-radius: 12px; padding: 2px 8px; font-size: 0.78rem;
        margin: 2px; font-weight: 500;
    }
    .score-badge {
        background: #1B4F8A; color: white; border-radius: 12px;
        padding: 3px 10px; font-size: 0.82rem; font-weight: 700;
    }
    .discovery-badge {
        background: #E67E22; color: white; border-radius: 12px;
        padding: 3px 10px; font-size: 0.82rem; font-weight: 700;
    }
    .cold-badge {
        background: #27AE60; color: white; border-radius: 12px;
        padding: 3px 10px; font-size: 0.82rem;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 700; color: #1B4F8A;
        border-bottom: 2px solid #1B4F8A; padding-bottom: 4px; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BM25
# ─────────────────────────────────────────────────────────────────────────────
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, corpus):
        self.docs_  = [doc.split() for doc in corpus]
        self.N_     = len(self.docs_)
        self.avgdl_ = np.mean([len(d) for d in self.docs_])
        df = defaultdict(int)
        for doc in self.docs_:
            for w in set(doc): df[w] += 1
        self.idf_ = {w: math.log((self.N_-n+0.5)/(n+0.5)+1) for w,n in df.items()}
        return self

    def transform(self, queries):
        scores = np.zeros((len(queries), self.N_))
        for qi, query in enumerate(queries):
            for di, doc in enumerate(self.docs_):
                dl, tf_map, s = len(doc), defaultdict(int), 0.0
                for w in doc: tf_map[w] += 1
                for t in query.split():
                    if t not in self.idf_: continue
                    tf = tf_map.get(t, 0)
                    s += self.idf_[t]*tf*(self.k1+1)/(tf+self.k1*(1-self.b+self.b*dl/self.avgdl_))
                scores[qi, di] = s
        return scores

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & MODEL (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    users_df   = pd.read_csv('users.csv').reset_index(drop=True)
    content_df = pd.read_csv('content.csv').reset_index(drop=True)
    return users_df, content_df

@st.cache_resource
def build_model(model_choice):
    users_df, content_df = load_data()

    # Preprocess
    for c in ['title','tags','type']: content_df[c] = content_df[c].fillna('')
    content_df['tags_clean'] = content_df['tags'].str.replace(',', ' ')
    content_df['combined_features'] = (
        content_df['title']+' '+content_df['title']+' '+
        content_df['tags_clean']+' '+content_df['type']
    ).str.lower()

    for c in ['interests','role','business_activity']: users_df[c] = users_df[c].fillna('')
    ic = users_df['interests'].str.replace(',', ' ')
    users_df['user_profile'] = (ic+' '+ic+' '+users_df['role']+' '+users_df['business_activity']).str.lower()

    corpus_content = list(content_df['combined_features'])
    corpus_users   = list(users_df['user_profile'])
    corpus_all     = corpus_content + corpus_users

    tfidf_vec = TfidfVectorizer(max_features=8000, ngram_range=(1,2),
                                stop_words='english', sublinear_tf=True)
    tfidf_vec.fit(corpus_all)

    if model_choice == 'TF-IDF':
        content_vecs = tfidf_vec.transform(corpus_content)
        user_vecs    = tfidf_vec.transform(corpus_users)
        sim_matrix   = cosine_similarity(user_vecs, content_vecs)

    elif model_choice == 'BM25':
        bm25 = BM25().fit(corpus_content)
        sim_raw    = bm25.transform(corpus_users)
        mx = sim_raw.max(axis=1, keepdims=True); mx[mx==0] = 1
        sim_matrix = sim_raw / mx
        content_vecs = tfidf_vec.transform(corpus_content)

    else:  # CountVectorizer
        from sklearn.feature_extraction.text import CountVectorizer
        count_vec = CountVectorizer(max_features=8000, ngram_range=(1,2), stop_words='english')
        count_vec.fit(corpus_all)
        content_vecs = count_vec.transform(corpus_content)
        user_vecs    = count_vec.transform(corpus_users)
        sim_matrix   = cosine_similarity(user_vecs, content_vecs)
        tfidf_vec    = count_vec   # use same interface for cold start

    return users_df, content_df, sim_matrix, tfidf_vec, tfidf_vec.transform(corpus_content)

def get_recs(user_idx, sim_matrix, content_df, top_n=10, diversity=0.2):
    scores     = sim_matrix[user_idx]
    sorted_idx = np.argsort(scores)[::-1]
    n_p  = int(top_n * (1 - diversity))
    n_d  = top_n - n_p
    top  = sorted_idx[:n_p]
    rest = sorted_idx[n_p:]
    disc = np.random.choice(rest, size=min(n_d, len(rest)), replace=False) \
           if n_d > 0 and len(rest) > 0 else []
    disc_set = set(disc)
    rows = []
    for rank, idx in enumerate(list(top)+list(disc), 1):
        p = content_df.iloc[idx]
        rows.append({'Rang': rank, 'Titre': p['title'], 'Tags': p['tags'],
                     'Type': p['type'], 'Score': round(float(scores[idx]), 4),
                     'idx': int(idx),
                     'Découverte': idx in disc_set})
    return pd.DataFrame(rows)

def cold_start_recs(interests, role, sector, tfidf_vec, content_vecs, content_df, top_n=10):
    profile = (interests+' '+interests+' '+role+' '+sector).lower()
    vec     = tfidf_vec.transform([profile])
    scores  = cosine_similarity(vec, content_vecs).flatten()
    top_idx = np.argsort(scores)[::-1][:top_n]
    rows    = []
    for rank, idx in enumerate(top_idx, 1):
        p = content_df.iloc[idx]
        rows.append({'Rang': rank, 'Titre': p['title'], 'Tags': p['tags'],
                     'Type': p['type'], 'Score': round(float(scores[idx]), 4),
                     'Découverte': False})
    return pd.DataFrame(rows)

def render_rec_card(row):
    tags_html = ''.join(
        f'<span class="tag-pill">{t.strip()}</span>'
        for t in str(row['Tags']).split(',')[:5]
    )
    badge = '<span class="discovery-badge">🔍 Découverte</span>' \
            if row['Découverte'] else '<span class="score-badge">✅ Personnalisé</span>'
    st.markdown(f"""
    <div class="rec-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <b style="font-size:1rem; color:#1B4F8A;">#{row['Rang']} — {row['Titre']}</b>
            <div>{badge} &nbsp; <span style="font-size:0.85rem;color:#888;">score: <b>{row['Score']}</b></span></div>
        </div>
        <div style="margin-top:6px;">{tags_html}</div>
        <div style="font-size:0.8rem;color:#999;margin-top:4px;">Type : {row['Type']}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
    st.markdown("## ⚙️ Configuration")

    model_choice = st.selectbox(
        "🧠 Modèle de vectorisation",
        ["TF-IDF", "BM25", "CountVectorizer"],
        index=0,
        help="TF-IDF = meilleur score global | BM25 = mieux pour textes courts | Count = baseline"
    )

    top_n = st.slider("📋 Nombre de recommandations", 3, 20, 8)

    diversity = st.slider(
        "🎲 Taux de découverte",
        0.0, 0.5, 0.2, 0.05,
        help="0% = 100% personnalisé | 50% = moitié aléatoire (anti-bulle de filtre)"
    )

    st.divider()
    st.markdown("### 📊 Dataset")
    users_df_raw, content_df_raw = load_data()
    st.metric("Utilisateurs", len(users_df_raw))
    st.metric("Posts", len(content_df_raw))
    st.divider()
    st.markdown("**Sources :**")
    st.markdown("- [LinkedIn Dataset — Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)")
    st.markdown("- [Medium Articles — Kaggle](https://www.kaggle.com/datasets/fabiochiusano/medium-articles)")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Chargement du modèle {model_choice}..."):
    users_df, content_df, sim_matrix, tfidf_vec, content_vecs = build_model(model_choice)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎯 Siratify — Système de Recommandation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Content-Based Filtering · TF-IDF / BM25 / CountVectorizer</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Utilisateur existant",
    "❄️ Cold Start",
    "🔗 Posts similaires",
    "📊 Évaluation & Comparaison"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXISTING USER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="section-header">Sélectionner un utilisateur</div>', unsafe_allow_html=True)
        uid_list   = users_df['user_id'].tolist()
        uid_labels = [f"{r['user_id']} — {r['full_name']} ({r['role']})"
                      for _, r in users_df.iterrows()]
        selected   = st.selectbox("Utilisateur", uid_labels, index=0)
        selected_uid = selected.split(' — ')[0]

        user_row = users_df[users_df['user_id'] == selected_uid].iloc[0]
        st.markdown(f"""
        <div class="metric-card">
            <b>👤 {user_row['full_name']}</b><br>
            💼 {user_row['role']}<br>
            🏢 {user_row['business_activity']}<br>
            ⭐ <i>{user_row['interests']}</i>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Recommandations personnalisées</div>', unsafe_allow_html=True)
        uidx = users_df[users_df['user_id'] == selected_uid].index[0]
        recs = get_recs(uidx, sim_matrix, content_df, top_n=top_n, diversity=diversity)

        for _, row in recs.iterrows():
            render_rec_card(row)

    # Score chart
    st.divider()
    st.markdown("#### 📊 Scores de similarité")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    recs_sorted = recs.sort_values('Score', ascending=True)
    colors_bar  = ['#E67E22' if d else '#1B4F8A' for d in recs_sorted['Découverte']]
    titles_short = [t[:35]+'...' if len(t)>35 else t for t in recs_sorted['Titre']]
    ax.barh(titles_short, recs_sorted['Score'], color=colors_bar, alpha=0.85)
    ax.set_xlabel('Score de similarité cosinus')
    ax.set_title(f'Scores pour {user_row["full_name"]} — Modèle {model_choice}', fontweight='bold')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#1B4F8A', label='Personnalisé'),
                        Patch(color='#E67E22', label='Découverte')])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COLD START
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    > **Cold Start** : nouvel utilisateur sans historique d'interactions.  
    > Le système utilise uniquement son profil déclaré pour recommander.
    """)

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown('<div class="section-header">Profil du nouvel utilisateur</div>', unsafe_allow_html=True)
        cs_interests = st.text_area(
            "⭐ Intérêts (séparés par des espaces)",
            value="machine learning python deep learning NLP data science",
            height=80
        )
        cs_role   = st.text_input("💼 Rôle / Poste", value="Data Scientist")
        cs_sector = st.selectbox("🏢 Secteur d'activité", [
            "Data & Analytics", "IT & Software Development",
            "Marketing & Communication", "Entrepreneurship & Management",
            "Finance & Banking", "Human Resources", "Design & Creative",
            "Sales & Business Development", "Education & Training", ""
        ])
        go_cs = st.button("🚀 Générer mes recommandations", use_container_width=True, type="primary")

    with col_b:
        if go_cs or True:
            st.markdown('<div class="section-header">Recommandations Cold Start</div>', unsafe_allow_html=True)
            cs_recs = cold_start_recs(cs_interests, cs_role, cs_sector,
                                       tfidf_vec, content_vecs, content_df, top_n=top_n)
            for _, row in cs_recs.iterrows():
                st.markdown(f"""
                <div class="rec-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <b style="color:#1B4F8A;">#{row['Rang']} — {row['Titre']}</b>
                        <span class="cold-badge">❄️ Cold Start</span>
                        <span style="font-size:0.85rem;color:#888;">score: <b>{row['Score']}</b></span>
                    </div>
                    <div style="margin-top:6px;">
                        {''.join(f'<span class="tag-pill">{t.strip()}</span>' for t in str(row["Tags"]).split(",")[:5])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SIMILAR POSTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("Trouver les posts les plus similaires à un post donné.")
    col_x, col_y = st.columns([1, 2])

    with col_x:
        post_labels = [f"{r['content_id']} — {r['title'][:50]}" for _, r in content_df.iterrows()]
        selected_post = st.selectbox("📄 Choisir un post de référence", post_labels, index=0)
        selected_cid  = selected_post.split(' — ')[0]
        ref_post = content_df[content_df['content_id'] == selected_cid].iloc[0]
        st.markdown(f"""
        <div class="metric-card">
            <b>{ref_post['title']}</b><br>
            <small>Tags : {ref_post['tags'][:80]}...</small>
        </div>
        """, unsafe_allow_html=True)
        n_similar = st.slider("Nombre de posts similaires", 3, 15, 6)

    with col_y:
        ref_idx  = content_df[content_df['content_id'] == selected_cid].index[0]
        sim_post = cosine_similarity(content_vecs[ref_idx], content_vecs).flatten()
        sim_post[ref_idx] = -1
        top_sim  = np.argsort(sim_post)[::-1][:n_similar]

        st.markdown('<div class="section-header">Posts similaires</div>', unsafe_allow_html=True)
        for rank, idx in enumerate(top_sim, 1):
            p = content_df.iloc[idx]
            tags_html = ''.join(f'<span class="tag-pill">{t.strip()}</span>'
                                for t in str(p['tags']).split(',')[:4])
            st.markdown(f"""
            <div class="rec-card">
                <div style="display:flex;justify-content:space-between;">
                    <b style="color:#1B4F8A;">#{rank} — {p['title']}</b>
                    <span class="score-badge">{round(float(sim_post[idx]),4)}</span>
                </div>
                <div style="margin-top:5px;">{tags_html}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 Métriques d'évaluation du modèle sélectionné")

    with st.spinner("Calcul des métriques (Precision / Recall / NDCG)..."):
        # Ground truth
        @st.cache_data
        def get_gt():
            rng = np.random.RandomState(42); gt = {}
            for _, user in users_df.iterrows():
                uid = user['user_id']
                interests = set(i.strip().lower() for i in str(user['interests']).split(','))
                sector    = user['business_activity'].lower()
                rel = [ci for ci, post in content_df.iterrows()
                       if interests & set(t.strip().lower() for t in str(post['tags']).split(','))
                       or sector in ' '.join(str(post['tags']).lower().split(','))]
                n_seen = rng.randint(5, max(6, min(21, len(rel)+2)))
                seen   = list(rng.choice(rel, size=min(n_seen, len(rel)), replace=False)) if rel else []
                non_rel = [i for i in range(len(content_df)) if i not in seen]
                n_noise = int(len(seen)*0.1)
                if n_noise > 0 and len(non_rel) >= n_noise:
                    seen += list(rng.choice(non_rel, size=n_noise, replace=False))
                gt[uid] = set(seen)
            return gt

        def prec_k(rec, rel, k): return len(set(rec[:k]) & rel) / k
        def rec_k(rec, rel, k):  return len(set(rec[:k]) & rel) / len(rel) if rel else 0.0
        def dcg_k(rec, rel, k):
            return sum(1.0/math.log2(i+1) for i,idx in enumerate(rec[:k],1) if idx in rel)
        def ndcg_k(rec, rel, k):
            if not rel: return 0.0
            idcg = sum(1.0/math.log2(i+1) for i in range(1, min(len(rel),k)+1))
            return dcg_k(rec, rel, k)/idcg if idcg > 0 else 0.0

        gt = get_gt()
        sample_uids = list(np.random.RandomState(42).choice(
            users_df['user_id'].tolist(), size=100, replace=False))

        ps, rs, ns = [], [], []
        for uid in sample_uids:
            uidx    = users_df[users_df['user_id']==uid].index[0]
            rec_idx = list(np.argsort(sim_matrix[uidx])[::-1][:10])
            rel     = gt.get(uid, set())
            ps.append(prec_k(rec_idx, rel, 10))
            rs.append(rec_k(rec_idx, rel, 10))
            ns.append(ndcg_k(rec_idx, rel, 10))

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision@10", f"{np.mean(ps):.4f}", delta="↑ vs baseline 0.10")
    col2.metric("Recall@10",    f"{np.mean(rs):.4f}", delta="↑ vs baseline 0.30")
    col3.metric("NDCG@10",      f"{np.mean(ns):.4f}", delta="↑ vs baseline 0.40")

    st.divider()
    st.markdown("#### Heatmap de similarité (20 users × 20 posts)")

    fig2, ax2 = plt.subplots(figsize=(16, 7))
    subset    = sim_matrix[:20, :20]
    u_labels  = [f"{r['full_name'].split()[0]} ({r['role'][:10]})"
                 for _, r in users_df.head(20).iterrows()]
    c_labels  = [t[:25]+'...' if len(t)>25 else t
                 for t in content_df['title'].tolist()[:20]]
    sns.heatmap(subset, xticklabels=c_labels, yticklabels=u_labels,
                cmap='YlOrRd', annot=True, fmt='.2f', linewidths=0.3,
                cbar_kws={'label': 'Score'}, ax=ax2)
    ax2.set_title(f'Matrice de Similarité [{model_choice}] — 20×20',
                  fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
