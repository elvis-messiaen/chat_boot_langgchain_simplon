# Chatbot RAG — Référentiel Développeur IA (version Ollama)

Version 100% locale du chatbot — aucune clé API requise.  
Le LLM (`llama3.2`) et les embeddings (`nomic-embed-text`) tournent entièrement via **Ollama** dans Docker.

## Stack technique

- **LangChain** — orchestration du pipeline RAG
- **Chroma** — base vectorielle locale (persistance sur disque)
- **Ollama** — LLM (`llama3.2`) et embeddings (`nomic-embed-text`) en local
- **LangGraph** — gestion de la mémoire de conversation
- **Streamlit** — interface utilisateur
- **PyPDF** — lecture du PDF du référentiel

## Prérequis

- Docker
- Docker Compose
- Aucune clé API OpenAI nécessaire

## Structure des fichiers spécifiques à cette version

```
.
├── main_ollama.py               # Code principal version Ollama
├── Dockerfile.ollama            # Image Docker de l'application Ollama
├── docker-compose.yml           # Orchestration app + service Ollama
└── chroma_langchain_db_ollama/  # Base vectorielle générée automatiquement
```

## Lancement

### 0. Arrêter la version OpenAI si elle tourne

Si le container OpenAI est actif, il occupe le port 8501 — arrête-le d'abord :

```bash
docker ps                        # récupérer l'ID du container OpenAI
docker stop <ID_DU_CONTAINER>
```

### 1. Builder et démarrer les containers

> Assure-toi que le container OpenAI est arrêté avant de lancer cette commande (port 8501 partagé).

```bash
docker compose -f docker-compose.yml up --build
```

Cela démarre deux containers :
- **ollama** — le service LLM local
- **chat-referentiel-ollama** — l'application Streamlit

### 2. Télécharger les modèles Ollama

**Obligatoire au premier lancement** — sans ces modèles l'application retourne une erreur `model not found`.

Dans un autre terminal, pendant que les containers tournent :

```bash
# Télécharger le modèle d'embeddings
docker exec -it ollama ollama pull nomic-embed-text

# Télécharger le LLM
docker exec -it ollama ollama pull llama3.2
```

> Ces téléchargements sont effectués une seule fois — les modèles sont persistés dans le volume `ollama_data`.  
> `llama3.2` ≈ 2 Go et `nomic-embed-text` ≈ 274 Mo — prévoir du temps selon ta connexion.

### 3. Accéder à l'application

```bash
http://localhost:8501
```

---

## Arrêter les containers

```bash
docker compose down
```

Pour supprimer également les volumes (modèles Ollama + base Chroma) :

```bash
docker compose down -v
```

---

## Variables d'environnement

Le fichier `.env` ne nécessite que :

```
FILE_PATH=data/Referentiel.pdf
```

La variable `OLLAMA_BASE_URL` est définie directement dans le `docker-compose.yml` et pointe automatiquement vers le service Ollama interne (`http://ollama:11434`).

---

## Fonctionnement du pipeline RAG

```
PDF → Découpe en chunks → Embeddings Ollama (nomic-embed-text) → Chroma
                                                                     ↓
Question utilisateur → Retriever (MMR, k=10) → Chunks pertinents
                                                                     ↓
                                    Prompt + Contexte → LLM Ollama (llama3.2) → Réponse
```

---

## Notes

- Au premier lancement, la base Chroma est générée automatiquement dans `chroma_langchain_db_ollama/`
- Pour forcer une réindexation, supprimer le dossier `chroma_langchain_db_ollama/` et relancer
- La mémoire de conversation repart à zéro à chaque redémarrage (stockage en RAM via LangGraph)
- Les modèles Ollama sont volumineux (`llama3.2` ≈ 2 Go, `nomic-embed-text` ≈ 274 Mo) — prévoir du temps au premier téléchargement
