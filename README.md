# 🎯 Siratify — Système de Recommandation IA (Content-Based Filtering)

> **Projet :** Moteur de recommandation de contenus professionnels basé sur les profils utilisateurs  
> **Approche :** Content-Based Filtering  
> **Dataset :** 500 utilisateurs · 300 posts

---

## 📌 C'est quoi ce projet ?

Siratify est un système de recommandation qui propose à chaque utilisateur des **posts pertinents** en fonction de son profil (intérêts, rôle, secteur d'activité). Il n'utilise **pas** les comportements d'autres utilisateurs (pas de Collaborative Filtering) — il se base uniquement sur la **similarité textuelle** entre le profil de l'utilisateur et le contenu des posts.

**Exemple concret :**  
Un Data Scientist intéressé par `machine learning, python, NLP` recevra en priorité des posts taggés avec ces mots-clés, car leur représentation vectorielle sera proche de son profil.

---

## 🗂️ Structure des fichiers

```
siratify/
├── siratify_recommendation_v2.ipynb   # Notebook principal (tout le pipeline)
├── users.csv                          # 500 utilisateurs avec leurs profils
├── content.csv                        # 300 posts avec titres, tags, types
└── app.py                             # Application (interface ou API)
```

### `users.csv` — colonnes principales
| Colonne | Description |
|---|---|
| `user_id` | Identifiant unique (ex: U0001) |
| `full_name` | Nom complet |
| `role` | Poste professionnel (ex: Data Scientist) |
| `interests` | Liste d'intérêts séparés par virgule |
| `business_activity` | Secteur d'activité (ex: Data & Analytics) |

### `content.csv` — colonnes principales
| Colonne | Description |
|---|---|
| `content_id` | Identifiant unique (ex: C0001) |
| `title` | Titre du post |
| `tags` | Mots-clés séparés par virgule |
| `type` | Type de contenu (article, video, podcast…) |

---

## ⚙️ Pipeline complet (cellule par cellule)

### 1. Imports & Configuration
Chargement des bibliothèques : `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`. Implémentation native de **BM25** sans dépendance externe.

### 2. Chargement des données
Lecture des fichiers `users.csv` et `content.csv`.

### 3. Analyse Exploratoire (EDA)
Visualisations : distribution des secteurs, top 10 des rôles, types de contenus, nombre d'intérêts par utilisateur → **fig1_eda.png**

### 4. Prétraitement
- **Posts** : concaténation `titre + titre + tags + type` (le titre est répété pour lui donner plus de poids)
- **Users** : concaténation `intérêts + intérêts + rôle + secteur`
- Tout est mis en minuscules

### 5. Vectorisation (3 modèles)
Chaque texte (profil user + post) est transformé en vecteur numérique. C'est ici que les 3 modèles divergent.

### 6. Simulation du ground truth
Un post est considéré **pertinent** pour un user si au moins un de ses intérêts apparaît dans les tags du post. On simule 5 à 20 interactions par user avec 10% de bruit aléatoire pour simuler des comportements réels imparfaits.

### 7. Métriques d'évaluation
Calcul de Precision@K, Recall@K, Coverage, Personalization et NDCG@K.

### 8. Comparaison des modèles ← *cellule corrigée*
Évaluation des 3 modèles avec K=5 et K=10, affichage des résultats en tableau.

### 9–17. Visualisations, recommandations, cold start, heatmap, résumé final

---

## 🔢 Les 3 modèles comparés

### TF-IDF (Term Frequency — Inverse Document Frequency)

**Principe :** Un mot est important s'il apparaît souvent dans un document ET rarement dans l'ensemble du corpus. Les mots très fréquents partout (ex: "the", "and") ont un poids faible même s'ils apparaissent souvent dans un post.

**Formule simplifiée :**
```
TF-IDF(mot, doc) = fréquence_dans_doc × log(N / nb_docs_avec_ce_mot)
```

**Paramètres utilisés ici :**
- `max_features=8000` — vocabulaire limité aux 8000 termes les plus fréquents
- `ngram_range=(1,2)` — prend en compte les mots seuls ET les bigrammes (ex: "machine learning")
- `sublinear_tf=True` — atténue les très hautes fréquences
- `stop_words='english'` — ignore les mots vides anglais

**Point fort :** Très efficace quand les mots-clés spécifiques au domaine sont discriminants.

---

### BM25 (Best Match 25)

**Principe :** Évolution du TF-IDF (Robertson et al., 1994). Il corrige deux biais du TF-IDF classique :
1. **Saturation TF** : au-delà d'un certain nombre de répétitions, un mot supplémentaire n'apporte plus d'information (contrôlé par `k1`)
2. **Normalisation par longueur** : un post long ne doit pas être favorisé juste parce qu'il contient plus de mots (contrôlé par `b`)

**Formule simplifiée :**
```
BM25(mot, doc) = IDF(mot) × (tf × (k1+1)) / (tf + k1 × (1 - b + b × longueur_doc/longueur_moyenne))
```

**Paramètres utilisés ici :**
- `k1=1.5` — contrôle la saturation de la fréquence
- `b=0.75` — contrôle la normalisation par longueur

**Important :** BM25 est implémenté **nativement** dans ce projet (classe `BM25` dans la cellule 1), sans bibliothèque externe. Les scores BM25 sont ensuite normalisés entre 0 et 1 pour être comparables aux autres modèles.

**Point fort :** Plus robuste que TF-IDF pour des documents de longueurs très variables.

---

### CountVectorizer

**Principe :** Le plus simple des 3. Compte juste la fréquence brute de chaque mot, sans aucune pondération IDF. Deux documents sont similaires s'ils partagent beaucoup de mots communs, peu importe leur rareté dans le corpus.

**Paramètres utilisés ici :**
- Mêmes que TF-IDF : `max_features=8000`, `ngram_range=(1,2)`, `stop_words='english'`
- Pas de `sublinear_tf` (pas de pondération)

**Point fort :** Très rapide, bon baseline. Utile quand les mots rares ne sont pas plus importants que les mots fréquents.

---

## 📐 Pourquoi comparer ces 3 modèles ?

Les 3 modèles répondent à la même question — *"quel post ressemble le plus à ce profil user ?"* — mais avec des philosophies différentes :

| | CountVectorizer | TF-IDF | BM25 |
|---|---|---|---|
| Pondération IDF | ❌ Non | ✅ Oui | ✅ Oui (+ saturation) |
| Normalisation longueur | ❌ Non | Partielle | ✅ Explicite |
| Complexité | Faible | Moyenne | Moyenne |
| Sensible aux mots rares | ❌ Non | ✅ Oui | ✅ Oui |

La comparaison permet de répondre à : *"Est-ce que la pondération sophistiquée de BM25 apporte réellement quelque chose sur ce dataset ?"* Dans certains cas, CountVectorizer performe mieux car les tags sont déjà des mots-clés sélectionnés — pas besoin de dé-pondérer les mots fréquents.

---

## 📊 Métriques d'évaluation

### Precision@K
*Parmi les K posts recommandés, combien sont vraiment pertinents ?*
```
Precision@K = |recommandés ∩ pertinents| / K
```

### Recall@K
*Parmi tous les posts pertinents pour cet user, combien sont dans le top K ?*
```
Recall@K = |recommandés ∩ pertinents| / |pertinents|
```

### NDCG@K (Normalized Discounted Cumulative Gain)
*Mesure de la qualité du classement : un post pertinent en position 1 vaut plus qu'en position 10.*
```
NDCG@K = DCG@K / IDCG@K     (normalisé entre 0 et 1)
```
C'est la métrique standard de la littérature sur les systèmes de recommandation.

### Coverage (%)
*Proportion du catalogue de posts qui sont recommandés à au moins un utilisateur.* Une couverture élevée = le système ne recommande pas toujours les mêmes posts.

### Personalization (%)
*Mesure à quel point les recommandations sont différentes entre utilisateurs.* Une personnalisation élevée = chaque user reçoit des recommandations vraiment adaptées.

### Score Composite
Combinaison pondérée finale pour désigner le meilleur modèle :
```
Score = 0.30 × Precision@10 + 0.25 × Recall@10 + 0.25 × NDCG@10
      + 0.10 × Coverage/100  + 0.10 × Personalization/100
```

---

## ❄️ Gestion du Cold Start

**Problème :** Un nouvel utilisateur n'a aucune interaction dans la base → pas de ligne dans la matrice de similarité.

**Solution :** On vectorise directement son profil déclaré (intérêts + rôle + secteur) avec le même modèle TF-IDF, puis on calcule la similarité cosinus contre tous les posts.

```python
recommender.recommend_cold_start(
    interests='machine learning python NLP',
    role='Data Scientist',
    sector='Data & Analytics',
    top_n=5
)
```

3 scénarios sont testés : profil complet (Data Scientist), profil complet (Marketing Manager), profil minimal (2 intérêts seulement).

---

## 🚀 Lancer le projet

```bash
# 1. Mettre users.csv, content.csv et le notebook dans le même dossier
# 2. Installer les dépendances
pip install pandas numpy scikit-learn matplotlib seaborn

# 3. Ouvrir le notebook
jupyter notebook siratify_recommendation_v2.ipynb

# 4. Run All (Kernel > Restart & Run All)
```

Les figures générées sont sauvegardées automatiquement :
- `fig1_eda.png` — Analyse exploratoire
- `fig2_comparaison_modeles.png` — Comparaison TF-IDF / BM25 / CountVectorizer
- `fig3_precision_recall_curves.png` — Courbes Precision@K et Recall@K
- `fig4_cold_start.png` — Utilisateur existant vs Cold Start
- `fig5_heatmap.png` — Matrice de similarité 20×20
- `fig6_summary.png` — Résumé comparatif final
- `fig7_ndcg.png` — Courbes NDCG@K

---

## 🏆 Conclusion

Le système permet de comparer objectivement 3 approches de vectorisation sur les mêmes données, les mêmes métriques et le même ground truth. Le **meilleur modèle** est sélectionné automatiquement par score composite et utilisé dans la classe `SiratifyRecommender` pour servir les recommandations finales — y compris pour les nouveaux utilisateurs via le mécanisme de cold start.
