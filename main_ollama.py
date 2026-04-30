from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List
import streamlit as st

# Chargement du .env — override=True force le rechargement si déjà chargé
load_dotenv(override=True)

# Récupération du chemin du PDF depuis le .env
FILE_PATH = os.getenv("FILE_PATH")

# URL du service Ollama — "ollama" est le nom du service dans docker-compose
# En local sans Docker, utiliser "http://localhost:11434"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Prompt augmenté personnalisé pour le référentiel DEV IA
PROMPT_TEMPLATE = {
    "system": """Tu es un expert en analyse du référentiel Développeur en Intelligence Artificielle.
    Tu identifies les compétences RNCP couvertes à partir du contexte fourni.
    Tu justifies toujours tes réponses avec un extrait du référentiel.
    Tu ne génères jamais d'hallucinations.

    Si la question porte directement sur le référentiel :
    Réponds précisément en te basant uniquement sur le contexte.

    Si l'utilisateur décrit un projet :
    - Liste des compétences couvertes
    - Extrait justificatif du référentiel
    - Liste des compétences non couvertes

    Synthétise ta réponse à partir des éléments du contexte disponibles, même partiels. Ne refuse jamais de répondre si le contexte contient des informations liées à la question.

    Contexte : {context}""",
    "human": "{input}"
}

# Chargement du PDF page par page
def lecture_pdf_PyPDFLoader(file_path):
    # Création du loader PyPDF en mode page — chaque page devient un Document séparé
    loader = PyPDFLoader(file_path, mode="page")
    # Chargement de toutes les pages du PDF
    documents = loader.load()
    return documents

# Découpe du texte en chunks pour le RAG
def decoupe_chunk():
    # Chargement du PDF via la fonction dédiée
    textes = lecture_pdf_PyPDFLoader(FILE_PATH)
    # Découpe en chunks de 300 caractères avec 200 de chevauchement pour ne pas perdre le contexte entre deux chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=200)
    # Application de la découpe sur les Documents chargés
    text_chunk = text_splitter.split_documents(textes)
    return text_chunk

# Génération des embeddings Ollama et stockage dans Chroma
def stock_embedding_chroma(text_chunk):
    # nomic-embed-text : meilleur modèle d'embeddings disponible sur Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    # Initialisation de la base vectorielle avec persistance locale
    vector_store = Chroma(
        # Nom de collection séparé de la version OpenAI pour éviter les conflits d'embeddings
        collection_name="referentiel_collection_ollama",
        # Fonction d'embedding Ollama
        embedding_function=embeddings,
        # Dossier de persistance séparé de la version OpenAI
        persist_directory="./chroma_langchain_db_ollama"
    )
    # Ajout des chunks vectorisés dans la base
    vector_store.add_documents(text_chunk)
    return vector_store

# Définition de l'état de la conversation gardé en mémoire par LangGraph
class EtatConversation(TypedDict):
    # Liste des messages échangés entre l'utilisateur et l'assistant
    messages: List
    # La question posée par l'utilisateur
    question: str

# Noeud du graphe : reçoit l'état, pose la question au RAG et met à jour l'historique
def noeud_rag(etat: EtatConversation):
    try:
        # Appel de la chaîne RAG avec la question et l'historique de conversation
        reponse = retrieval_chain.invoke({
            "input": etat["question"],
            "chat_history": etat["messages"]
        })
        # Ajout de la question et de la réponse dans l'historique des messages
        return {"messages": etat["messages"] + [
            HumanMessage(content=etat["question"]),
            AIMessage(content=reponse["answer"])
        ]}
    except Exception as e:
        # En cas d'erreur LLM ou Ollama indisponible, retourne un message d'erreur
        return {"messages": etat["messages"] + [
            HumanMessage(content=etat["question"]),
            AIMessage(content=f"Le service est temporairement indisponible : {e}")
        ]}

# Création de la chaîne RAG réutilisable par le noeud
def construire_retrieval_chain(vector_store):
    # Transforme Chroma en "chercheur" : reçoit une question, retourne les chunks les plus proches
    # MMR (Maximum Marginal Relevance) force la diversité des chunks retournés
    # fetch_k=30 : Chroma récupère 30 candidats, MMR sélectionne les 10 plus diversifiés
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 30}
    )

    # Modèle du message envoyé au LLM :
    # - prompt système (rôle d'expert jury + {context} injecté automatiquement avec les chunks trouvés)
    # - {chat_history} : historique récupéré depuis MemorySaver via etat["messages"] dans noeud_rag
    # - question de l'utilisateur
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_TEMPLATE["system"]),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    # Prend les chunks trouvés par Chroma et les injecte dans {context} du prompt
    combine_docs_chain = create_stuff_documents_chain(model, retrieval_qa_chat_prompt)

    # Chaîne finale : question → Chroma → chunks → prompt → LLM → réponse
    return create_retrieval_chain(retriever, combine_docs_chain)

# Création du graphe LangGraph avec mémoire persistante
def memoire_LangGraph():
    # Construction du graphe avec un seul noeud RAG
    graphe = StateGraph(EtatConversation)
    # Ajout du noeud RAG au graphe
    graphe.add_node("rag", noeud_rag)
    # Définition du noeud d'entrée
    graphe.set_entry_point("rag")
    # Connexion du noeud RAG à la fin du graphe
    graphe.add_edge("rag", END)

    # Mémoire persistante entre les appels pour conserver l'historique
    memoire = MemorySaver()
    # Compilation du graphe avec le checkpointer mémoire
    return graphe.compile(checkpointer=memoire)

# Initialisation du LLM Ollama — llama3.2 tourne localement via le service Ollama
model = ChatOllama(model="llama3.2", base_url=OLLAMA_BASE_URL)

# Variable globale pour rendre retrieval_chain accessible depuis noeud_rag
retrieval_chain = None

# Fonction d'ajout de fichier — accepte .py, .pdf, .txt, .md via drag and drop Streamlit
def ajout_fichier():
    # Widget Streamlit de dépôt de fichier — drag and drop natif
    fichier = st.file_uploader(
        "Ajouter un fichier (optionnel)",
        type=["py", "pdf", "txt", "md"],
        label_visibility="collapsed"
    )

    # Si aucun fichier déposé, on retourne None
    if fichier is None:
        return None

    # Traitement spécifique pour les PDF — extraction du texte page par page
    if fichier.type == "application/pdf":
        import pdfplumber
        with pdfplumber.open(fichier) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    # Pour .py, .txt, .md — lecture directe du contenu texte
    return fichier.read().decode("utf-8")

# Fonction principale qui orchestre toutes les étapes et lance le chatbot
def mise_en_fonction():
    # retrieval_chain est déclarée globalement pour être accessible depuis noeud_rag
    # sans la passer en paramètre à chaque appel LangGraph
    global retrieval_chain

    # Titre affiché en haut de l'interface Streamlit
    st.title("Interface de question sur le référentiel : développeur IA (Ollama)")

    # Initialisation de l'historique des questions et du compteur de clé au premier chargement
    if "historique" not in st.session_state:
        st.session_state.historique = []
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0
    # Initialisation du contexte sélectionné — None si aucune réponse n'a été choisie pour continuer
    if "contexte_selectionne" not in st.session_state:
        st.session_state.contexte_selectionne = None

    # Champ de saisie — la key dynamique permet de vider le champ après soumission
    # on_change=lambda: None force Streamlit à recharger le script à chaque frappe
    question = st.text_area("Quelle est votre question sur le référentiel ?", key=f"question_input_{st.session_state.input_key}", height=200, on_change=lambda: None)

    # Bouton grisé si le champ est vide, actif dès qu'une lettre est tapée
    analyser = st.button("Analyser", disabled=not bool(question))

    # Appel du widget d'upload — retourne le contenu du fichier ou None si aucun fichier déposé
    contenu_fichier = ajout_fichier()

    # Chemin du dossier de persistance Chroma Ollama — séparé de la version OpenAI
    CHROMA_DIR = "./chroma_langchain_db_ollama"

    # nomic-embed-text : même modèle qu'à l'indexation pour que les vecteurs soient compatibles
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)

    # Si le dossier Chroma n'existe pas : on indexe le PDF (découpe + embeddings + stockage)
    # Si le dossier existe déjà : on recharge la base existante pour éviter les doublons
    try:
        if not os.path.exists(CHROMA_DIR):
            # Premier lancement : découpe et indexation du PDF
            chunks = decoupe_chunk()
            vector_store = stock_embedding_chroma(chunks)
        else:
            # Reconnexion à la base vectorielle déjà persistée sur disque
            vector_store = Chroma(
                collection_name="referentiel_collection_ollama",
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR
            )
    except Exception as e:
        # Affichage de l'erreur dans Streamlit et arrêt de la fonction
        st.error(f"Erreur d'initialisation de la base vectorielle : {e}")
        return

    # Branche Chroma + prompt + LLM ensemble et stocke dans la variable globale
    # pour que noeud_rag puisse l'appeler via retrieval_chain.invoke(...)
    retrieval_chain = construire_retrieval_chain(vector_store)

    # Compile le graphe LangGraph avec le MemorySaver pour conserver l'historique entre les appels
    app = memoire_LangGraph()

    # thread_id identifie la session : même thread_id = même historique retrouvé dans MemorySaver
    # changer le thread_id repart avec une mémoire vide
    config = {"configurable": {"thread_id": "session_ollama"}}

    # Affichage du contexte sélectionné si l'utilisateur a cliqué sur "Continuer cette conversation" Cela mets une partie du context en affichage. Mais tous le context et bien pris en compte
    if st.session_state.contexte_selectionne:
        st.info(f"Contexte sélectionné : {st.session_state.contexte_selectionne[:150]}...")

    # Déclenchement uniquement si le bouton est cliqué et que la question n'est pas vide
    if analyser and question:
        # Si un contexte a été sélectionné, on l'injecte dans la question pour que le LLM en tienne compte
        question_complete = question
        if st.session_state.contexte_selectionne:
            question_complete = f"Contexte de la réponse précédente : {st.session_state.contexte_selectionne}\n\nNouvelle question : {question}"

        # Si un fichier a été déposé, on ajoute son contenu à la question pour enrichir le contexte
        # Troncature à 3000 caractères pour ne pas dépasser la fenêtre de contexte du LLM
        if contenu_fichier:
            contenu_tronque = contenu_fichier[:3000]
            question_complete += f"\n\nContenu du fichier fourni :\n{contenu_tronque}"

        # Spinner affiché pendant le traitement pour indiquer que la requête est en cours
        with st.spinner("Analyse de votre demande en cours..."):
            # Invocation du graphe LangGraph avec la question enrichie du contexte si disponible
            reponse = app.invoke(
                {"question": question_complete, "messages": []},
                config=config
            )
        # Ajout de la question et de la réponse dans l'historique de session
        st.session_state.historique.append({
            "question": question,
            "reponse": reponse["messages"][-1].content
        })
        # Réinitialisation du contexte sélectionné après utilisation
        st.session_state.contexte_selectionne = None
        # Incrémentation de la key pour vider le champ de saisie
        st.session_state.input_key += 1
        # Rechargement de la page pour afficher le champ vide
        st.rerun()

    # Affichage de l'historique sous forme de blocs cliquables
    # le dernier échange s'ouvre automatiquement, les anciens restent repliés
    for i, echange in enumerate(st.session_state.historique):
        with st.expander(f"Question {i+1}", expanded=(i == len(st.session_state.historique) - 1)):
            # Affichage de la question posée par l'utilisateur
            st.markdown(f"**{echange['question']}**")
            # Affichage de la réponse à l'intérieur du bloc déplié
            st.write(echange['reponse'])
            # Bouton pour sélectionner cette réponse comme contexte de la prochaine question
            if st.button(f"Continuer cette conversation", key=f"continuer_{i}"):
                # Stockage de la réponse sélectionnée dans session_state
                st.session_state.contexte_selectionne = echange['reponse']
                # Rechargement pour afficher le bandeau de contexte sélectionné
                st.rerun()

# Lancement de l'application
mise_en_fonction()
