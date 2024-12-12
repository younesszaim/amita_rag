# AmitaGPT : Assistant Interactif de Recherche Documentaire

AmitaGPT est une application intelligente de recherche et de réponse documentaire, développée en français, utilisant Streamlit, LangChain et les modèles de langage OpenAI. 

L'application permet aux utilisateurs d'interagir avec une base de documents vectorielle et d'obtenir des réponses contextuellement pertinentes 
avec  une journalisation pour suivre les performances et déboguer les problèmes.

### Démonstration
![amitaGPT POC.mov](demo/amitaGPT%20POC.mov)

## Features
- 🇫🇷 Support natif en français
- 🔍 Recherche documentaire avancée avec MultiQueryRetriever
- 💬 Interface de chat interactive
- 📄 Base de données vectorielle pour une recherche documentaire efficace
- 🚀 Génération de réponses en temps réel

## Prérequis 

- Python 3.8+
- Clé API OpenAI
- Bibliothèques :
  - streamlit
  - langchain
  - python-dotenv
  - faiss-cpu
  - openai

## Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/votre-nom-utilisateur/amita-gpt.git
cd amita-gpt
```

2. Run the system:
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4.Configurez les variables d'environnement :
Créez un fichier .config à la racine du projet et ajoutez votre clé API OpenAI :
```bash
OPENAI_API_KEY=votre_clé_openai_ici
```

## Utilisation

- Composants principaux :
  - `InteractiveRAG`: Classe principale gérant la recherche documentaire et la génération de réponses
  - `_load_or_create_vector_db()`: Crée ou charge une base de données vectorielle FAISS
  - `run_rag_prompt()`:  Génère des réponses contextuelles basées sur la recherche documentaire
  - Interface Streamlit pour un chat interactif

## Usage

```bash
streamlit run ./main.py
```

# Licence
[Ajoutez une licence ici]

# Contributeurs
- Yoann Deluchat
- Thiziri Sassi
- Julie Massé
- Youness ZAIM