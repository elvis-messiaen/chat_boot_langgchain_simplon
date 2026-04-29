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

## Lancement avec Ollama (en cours d'implémentation)

> Section à compléter après mise en place du `docker-compose.yml`
