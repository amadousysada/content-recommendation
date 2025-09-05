# My Content - Système de Recommandation d'Articles

<div align="left">

[![Deploy Functions (remote build via publish-profile)](https://github.com/amadousysada/content-recommendation/actions/workflows/deploy-functions.yml/badge.svg)](https://github.com/amadousysada/content-recommendation/actions/workflows/deploy-functions.yml) [![Deploy Streamlit to Azure Web App](https://github.com/amadousysada/content-recommendation/actions/workflows/deploy-front.yml/badge.svg)](https://github.com/amadousysada/content-recommendation/actions/workflows/deploy-front.yml)

</div>

##  Vue d'ensemble

**My Content** est une start-up qui encourage la lecture en recommandant des contenus pertinents pour ses utilisateurs. Ce projet implémente un MVP de système de recommandation d'articles utilisant une architecture serverless Azure Functions avec une interface Streamlit.

###  Fonctionnalité Principale
> "En tant qu'utilisateur de l'application, je vais recevoir une sélection de cinq articles."

##  Architecture Technique

### Choix d'Architecture
Nous avons opté pour la **deuxième architecture proposée** par Julien :
- **Backend** : Azure Functions (serverless) avec intégration directe Azure Blob Storage
- **Frontend** : Interface Streamlit locale
- **Stockage** : Azure Blob Storage pour données et modèles

### Diagramme d'Architecture
```
┌─────────────────┐    HTTP/REST    ┌──────────────────┐
│   Streamlit     │ ──────────────► │  Azure Functions │
│   (Interface)   │                 │   (Backend)      │
└─────────────────┘                 └──────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │ Azure Blob       │
                                    │ Storage          │
                                    │ - models/        │
                                    │ - data/          │
                                    │ - precomputed/   │
                                    └──────────────────┘
```

## Système de Recommandation

### Approche Hybride

#### 1. **Content-Based Recommender** (Principal)
- **Algorithme** : Similarité cosinus sur embeddings d'articles
- **Profil utilisateur** : Moyenne pondérée des articles cliqués
- **Avantage** : Recommandations personnalisées basées sur le contenu

#### 2. **Popularity Fallback** (Cold-start)
- **Déclenchement** : Utilisateurs sans historique de clics
- **Algorithme** : Articles les plus populaires non vus
- **Avantage** : Gestion du problème du démarrage à froid

### Flux de Recommandation
```
Utilisateur → Historique clics → Profil utilisateur → Similarité cosinus → Top-N articles
     ↓
Si pas d'historique → Articles populaires → Top-N articles
```

##  Gestion des Données

### Structure des Données
- **articles_metadata.csv** : Métadonnées des articles (ID, catégorie, date, éditeur, mots)
- **clicks_sample.csv** : Interactions utilisateurs (clics, sessions, contexte)
- **embeddings.pkl** : Vecteurs d'articles pré-entraînés

### Optimisations
- **Cache intelligent** : Mise en cache des DataFrames en mémoire
- **Pré-calcul** : Recommandations pré-calculées stockées dans Blob Storage
- **Parsing flexible** : Support de multiples formats d'embeddings

## API Endpoints

### `GET /api/recommend_get`
**Paramètres :**
- `user_id` (requis) : ID de l'utilisateur
- `topn` (optionnel) : Nombre de recommandations (défaut: 10)
- `verbose` (optionnel) : Affichage détaillé (défaut: false)
- `realtime` (optionnel) : Calcul en temps réel vs cache (défaut: false)

**Réponse :**
```json
{
  "user_id": 50,
  "source": "realtime-sklearn",
  "model": "Content-Based",
  "recommendations": [
    {
      "article_id": 12345,
      "score": 0.85,
      "category_id": 12,
      "created_at_ts": 1640995200000,
      "publisher_id": 5,
      "words_count": 1200
    }
  ]
}
```

### `POST /api/precompute_http`
Pré-calcule les recommandations pour tous les utilisateurs et les stocke dans Blob Storage.

### `GET /api/diag_model`
Diagnostic de l'état des modèles et données.

## Installation et Déploiement

### 🌐 Application en Ligne
**Démo disponible** : [http://reco-streamlit-43896.azurewebsites.net/](http://reco-streamlit-43896.azurewebsites.net/)

### Prérequis
- Python 3.11+
- Azure Functions Core Tools
- Compte Azure avec Blob Storage

### Configuration Locale
1. **Cloner le repository**
2. **Configurer Azure Blob Storage** dans `app/local.settings.json`
3. **Installer les dépendances** :
   ```bash
   cd app && pip install -r requirements.txt
   cd ../streamlit && pip install -r requirements.txt
   ```
4. **Lancer l'application** :
   ```bash
   python start_apps.py
   ```

### Variables d'Environnement
```json
{
  "AzureWebJobsStorage": "connection_string_azure_storage",
  "BLOB_CONTAINER_MODELS": "models",
  "BLOB_CONTAINER_DATA": "data", 
  "BLOB_CONTAINER_PRECOMPUTED": "precomputed"
}
```

## Évolutivité et Architecture Cible

### Gestion des Nouveaux Utilisateurs
- **Cold-start** : Système de recommandations populaires
- **Apprentissage progressif** : Construction du profil au fil des interactions
- **Cache adaptatif** : Pré-calcul des recommandations fréquentes

### Gestion des Nouveaux Articles
- **Architecture modulaire** : Ajout facile de nouveaux embeddings
- **Mise à jour des modèles** : Via Azure Blob Storage
- **Parsing flexible** : Support de différents formats de données

### Optimisations Futures
- **Réduction de dimension** : PCA pour optimiser les embeddings volumineux
- **Mise à jour incrémentale** : Recalcul partiel des recommandations
- **Monitoring** : Métriques de performance et qualité des recommandations

## Technologies Utilisées

### Backend
- **Azure Functions** : Serverless computing
- **Azure Blob Storage** : Stockage des données et modèles
- **Python 3.11** : Langage principal
- **scikit-learn** : Calculs de similarité cosinus
- **pandas/numpy** : Manipulation des données

### Frontend
- **Streamlit** : Interface utilisateur web
- **requests** : Communication avec l'API

### Développement
- **Jupyter Notebook** : Analyse exploratoire des données
- **Azure Functions Core Tools** : Développement local
