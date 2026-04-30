# Chatbot RAG — Référentiel Développeur IA

Chatbot conversationnel basé sur le référentiel RNCP **Développeur en Intelligence Artificielle**.  
Il permet d'interroger le référentiel en langage naturel et d'identifier les compétences couvertes par un projet.

## Stack technique

- **LangChain** — orchestration du pipeline RAG
- **Chroma** — base vectorielle locale (persistance sur disque)
- **OpenAI** — embeddings (`text-embedding-3-large`) et LLM (`gpt-4o-mini`)
- **LangGraph** — gestion de la mémoire de conversation
- **Streamlit** — interface utilisateur
- **PyPDF** — lecture du PDF du référentiel

## Prérequis

- Python 3.12
- Docker
- Une clé API OpenAI

## Structure du projet

```
.
├── main.py                  # Code principal
├── requirements.txt         # Dépendances Python
├── Dockerfile               # Image Docker de l'application
├── .env                     # Variables d'environnement (non versionné)
├── .dockerignore            # Fichiers exclus du container
├── data/
│   └── Referentiel.pdf      # PDF du référentiel DEV IA
└── chroma_langchain_db/     # Base vectorielle générée automatiquement
```

## Installation et lancement en local (sans Docker)

**1. Créer et activer l'environnement virtuel**

```bash
python -m venv .venv
source .venv/bin/activate
```

**2. Installer les dépendances**

```bash
pip install -r requirements.txt
```

**3. Créer le fichier `.env`**

```
OPENAI_API_KEY=sk-...
FILE_PATH=data/Referentiel.pdf
```

**4. Lancer l'application**

```bash
streamlit run main.py
```

L'application est accessible sur `http://localhost:8501`

---

## Lancement avec Docker

### Builder l'image

```bash
docker build -t chat-referentiel .
```

### Lancer le container

```bash
docker run --env-file .env -p 8501:8501 chat-referentiel
```

L'application est accessible sur `http://localhost:8501`

> Si le port 8501 est déjà utilisé (ex: Streamlit local en cours) :
> ```bash
> docker run --env-file .env -p 8502:8501 chat-referentiel
> ```
> Puis ouvrir `http://localhost:8502`

### Arrêter le container

```bash
docker ps                        # récupérer l'ID du container
docker stop <ID_DU_CONTAINER>
```

> Si tu veux ensuite lancer la version Ollama (`docker compose up`), arrête d'abord ce container — les deux utilisent le port 8501.

---

## Utilisation

L'interface propose un champ de saisie en bas de page.

**Questions directes sur le référentiel :**
> "Quels sont les blocs de compétences ?"
> "Quelles sont les compétences du bloc 2 ?"
> "Quelles sont les modalités d'évaluation ?"

**Description d'un projet :**
> "J'ai réalisé un projet de collecte de données avec une API REST et stockage en base SQL..."

Le chatbot retourne alors :
- La liste des compétences RNCP couvertes
- Un extrait justificatif du référentiel
- La liste des compétences non couvertes

---

## Fonctionnement du pipeline RAG

```
PDF → Découpe en chunks → Embeddings → Chroma
                                          ↓
Question utilisateur → Retriever (MMR, k=10) → Chunks pertinents
                                          ↓
                              Prompt + Contexte → LLM → Réponse
```

1. **Indexation** (une seule fois) : le PDF est découpé en chunks de 300 caractères avec un overlap de 200, vectorisé et stocké dans Chroma
2. **Requête** : la question est vectorisée, Chroma retourne les 10 chunks les plus pertinents via MMR
3. **Génération** : les chunks sont injectés dans le prompt, le LLM génère la réponse
4. **Mémoire** : LangGraph conserve l'historique de la conversation via MemorySaver

---

## Variables d'environnement

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Clé API OpenAI |
| `FILE_PATH` | Chemin vers le PDF du référentiel |

---

## Notes

- La base Chroma est générée automatiquement au premier lancement si le dossier `chroma_langchain_db` n'existe pas
- Pour forcer une réindexation, supprimer le dossier `chroma_langchain_db` et relancer
- La mémoire de conversation (LangGraph) est en RAM — elle repart à zéro à chaque redémarrage

---
---

## Déploiement sur Azure (App Service)

### Prérequis
- Azure CLI installé : `brew install azure-cli`
- Docker Desktop installé et lancé
- Un compte Azure avec un groupe de ressources disponible
- Ta clé API OpenAI

### Étape 1 — Connexion à Azure

```bash
az login
```

Un navigateur s'ouvre, connecte-toi avec ton compte Azure. Appuie sur Entrée pour valider la subscription par défaut.

### Étape 2 — Créer un Container Registry dans le portail Azure

1. Dans la barre de recherche Azure, tape **"Container Registry"**
2. Clique sur **"Container registries"** → **"+ Créer"**
3. Remplis :
   - Groupe de ressources : ton groupe existant
   - Nom du registre : `<tonnom>chatbot` (alphanumérique uniquement, 5-50 caractères)
   - Région : France Central
   - SKU : Basic
   - Mode d'autorisation : **Autorisations du Registre RBAC**
4. Clique sur **"Vérifier + créer"** puis **"Créer"**

### Étape 3 — Récupérer les identifiants du registry

1. Dans ton Container Registry → **"Clés d'accès"**
2. Active **"Utilisateur administrateur"**
3. Note : le **serveur de connexion**, le **nom d'utilisateur** et le **mot de passe**

### Étape 4 — Connecter Docker au registry

```bash
docker login <serveur-de-connexion>
```

Saisis le nom d'utilisateur et le mot de passe notés à l'étape précédente.

### Étape 5 — Builder l'image (AMD64 obligatoire sur Apple Silicon)

```bash
docker build --platform linux/amd64 -t <serveur-de-connexion>/chatbot-referentiel:latest .
```

> **Important** : le flag `--platform linux/amd64` est obligatoire sur Mac Apple Silicon (M1/M2/M3).
> Sans ce flag, l'image est ARM et Azure retourne une erreur `exec format error`.

### Étape 6 — Pousser l'image vers Azure

```bash
docker push <serveur-de-connexion>/chatbot-referentiel:latest
```

### Étape 7 — Créer l'App Service dans le portail Azure

1. Dans la barre de recherche Azure, tape **"App Services"**
2. Clique sur **"App Services"** → **"+ Créer"** → **"Application web"**
3. Remplis :
   - Groupe de ressources : ton groupe
   - Nom : `chatbot-referentiel`
   - Publier : **Conteneur**
   - Système d'exploitation : **Linux**
   - Région : **France Central**
   - Plan de tarification : **B1** (le moins cher compatible Docker, ~13 USD/mois)
4. Clique sur **"Vérifier + créer"** puis **"Créer"**

### Étape 8 — Configurer le conteneur

1. Dans l'App Service → **"Centre de déploiement"**
2. Clique sur **"main"** dans la liste
3. Remplis :
   - Source de l'image : **Azure Container Registry**
   - Registre : ton registry
   - Authentification : **Informations d'identification de l'administrateur**
   - Image : `chatbot-referentiel`
   - Balise d'image : `latest`
   - Port : **8501**
   - Commande de démarrage : laisser vide
4. Clique sur **"Appliquer"**

### Étape 9 — Configurer les variables d'environnement

1. Dans l'App Service → **"Paramètres"** → **"Variables d'environnement"**
2. Ajoute ces deux variables via **"+ Ajouter"** :

| Variable | Valeur |
|---|---|
| `OPENAI_API_KEY` | Ta clé API OpenAI |
| `FILE_PATH` | `data/Referentiel.pdf` |

3. Clique sur **"Appliquer"** puis **"Confirmer"**

### Étape 10 — Redémarrer et accéder à l'application

1. Dans l'App Service → **"Vue d'ensemble"** → **"Redémarrer"**
2. Attends 2 à 5 minutes au premier démarrage (téléchargement de l'image)
3. Clique sur le **"Domaine par défaut"** pour ouvrir l'application

En cas d'erreur au démarrage, consulte les logs dans **"Supervision"** → **"Flux de journaux"**.

### Mettre à jour l'application après modification du code

```bash
docker build --platform linux/amd64 -t <serveur-de-connexion>/chatbot-referentiel:latest .
docker push <serveur-de-connexion>/chatbot-referentiel:latest
```

Puis dans le portail Azure → App Service → **"Redémarrer"**.
