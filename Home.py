import os

import streamlit as st
import time
import streamlit_shadcn_ui as ui
from main import ExpertiseRAG, BusinessRAG, InterneRAG, EssentielsRAG
import streamlit_antd_components as sac
st.set_page_config(layout="wide", page_icon='pages/logo/img.png', initial_sidebar_state="collapsed")
st.logo(image='pages/logo/logo.png',size="large")
# --- Load CSS ---
def load_css_file(css_file_path):
    with open(css_file_path) as f:
        return st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css_file('pages/main.css')
# st.markdown("""
#     <style>
#     .stSidebar{
#         background-color: #FFFFFF;
#     }
#     .stSidebarContent{
#         color: #000000;
#     }
#     </style>
# """, unsafe_allow_html=True)

expertise_rag = ExpertiseRAG()
business_rag = BusinessRAG()
essentiels_rag = EssentielsRAG()
interne_rag = InterneRAG()
st.session_state.rag = ''
st.session_state.data_path = ''
# --- Helper Functions ---
def add_vertical_space(height):
    st.markdown(f'<div style="height: {height}px;"></div>', unsafe_allow_html=True)

# # --- Logo and Title ---
logo = 'pages/logo/logo.png'
col1, col2 = st.columns([1,2])
with col1:
    st.image(logo, width=800)

with col2:
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h1 style="margin-top: 10px;">Amita GPT</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="text-align: center;">
            <h3 style="margin-top: 10px;">
            Bienvenue sur AMITA GPT, votre assistant intelligent dédié à l'expertise, aux missions, et à l'organisation d'Amita Conseil. Accédez en un instant à toutes les informations clés sur nos offres, nos consultants et nos réussites.</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
add_vertical_space(50)

cols = st.columns(4)
with cols[0]:
    with st.container(border=True, key='Expertise'):
        add_vertical_space(10)
        st.markdown("### Expertise")
        #st.markdown("##### Join and Contribute")
        st.markdown("""
Les expertises d’Amita Conseil et les domaines d’intervention ainsi que les offres de conseil proposées. """)
        add_vertical_space(10)
        # expertise = sac.buttons([sac.ButtonsItem(label='expertise', icon='google', color='#25C3B0')],
        #             align='center')
        col1, col2, col3 = st.columns(3)
        with col2:
            expertise = st.button(label = 'Explore')
            add_vertical_space(5)

with cols[1]:
    with st.container(border=True, key='Business'):
        add_vertical_space(10)
        st.markdown("### Business")
        #st.markdown("##### Join and Contribute")
        st.markdown("""
Les missions réalisées par Amita Conseil, les expertises mobilisées, ainsi que les clients accompagnés.""")
        add_vertical_space(10)
        # business= sac.buttons([sac.ButtonsItem(label='Business', icon='google', color='#25C3B0')], align='center',
        #                        )
        col1,col2,col3 = st.columns(3)
        with col2:
            business = st.button(label= 'Explore', key='test')
            add_vertical_space(5)

with cols[2]:
    with st.container(border=True, key='Interne'):
        add_vertical_space(10)
        st.markdown("### Interne")
        #st.markdown("##### Join and Contribute")
        st.markdown("""
Les activités internes d’Amita, notamment les groupes de travail (GT), les initiatives internes, et les projets collaboratifs.""")
        add_vertical_space(10)
        #other = sac.buttons([sac.ButtonsItem(label='Other', icon='google', color='#25C3B0')], align='center')
        col1, col2, col3 = st.columns(3)
        with col2:
            interne = st.button(label = 'Explore', key='test3')
            add_vertical_space(5)

with cols[3]:
    with st.container(border=True, key='Essentiel'):
        add_vertical_space(10)
        st.markdown("### Essentiel")
        #st.markdown("##### Join and Contribute")
        st.markdown("""
L’organisation interne d’Amita, incluant la présentation institutionnelle, et les structures internes.""")
        add_vertical_space(10)
        #other = sac.buttons([sac.ButtonsItem(label='Other', icon='google', color='#25C3B0')], align='center')
        col1, col2, col3 = st.columns(3)
        with col2:
            essentiel = st.button(label = 'Explore', key='test2')
            add_vertical_space(5)

if business:
    st.session_state.rag = business_rag
    st.session_state.data_path = os.getenv('DATA_PATH_BUSINESS')
    st.switch_page('pages/Business.py')

if expertise:
    st.session_state.rag = expertise_rag
    st.session_state.data_path = os.getenv('DATA_PATH_EXPERTISE')
    st.switch_page('pages/Expertise.py')

if interne:
    st.session_state.rag = interne_rag
    st.session_state.data_path = os.getenv('DATA_PATH_INTERNE')
    st.switch_page('pages/Interne.py')

if essentiel:
    st.session_state.rag = essentiels_rag
    st.session_state.data_path = os.getenv('DATA_PATH_ESSENTIELS')
    st.switch_page('pages/Essentiel.py')

# expertise_rag = ExpertiseRAG()
# business_rag = BusinessRAG()
#  # Display the logo at the top
# st.image("./image/img.png", width=200)
# # Initialize session state for chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
#
# # Input field for user question
# question = st.text_input("Wellcome to AmitaGPT! Comment puis-je t'aider ? 😊", "")
# exp_button = st.button("Expertise")
# business_button = st.button("Business")
#
# # When the "Réponse" button is clicked
# if st.button("Réponse"):
#     pass
#     if question.strip():
#         # Add user's question to chat history
#         st.session_state.chat_history.append({"role": "user", "message": question})
#
#         # Display a progress bar
#         with st.spinner('Génération de la réponse...'):
#             progress_bar = st.progress(0)
#
#             # Simulate response generation process
#             for i in range(10):
#                 time.sleep(0.1)  # Simulate time taken to generate response
#                 progress_bar.progress((i + 1) * 10)
#
#             # Generate answer using the RAG system
#             result = business_rag.run_rag_prompt(question=question)
#             answer = result["response"]
#             resources = result["resources"]
#
#             # Add assistant's answer to chat history
#             st.session_state.chat_history.append(
#                 {"role": "assistant", "message": answer, "resources": resources})
#
# # Display chat history in reverse order (latest first)
# st.write("### Conversation :")
# pass
# for chat in reversed(st.session_state.chat_history):
#     if chat["role"] == "user":
#         st.markdown(f"**Vous** : {chat['message']}")
#     else:
#         st.markdown(f"**AmitaGPT** : {chat['message']}")
#         if "resources" in chat:
#             st.markdown("**Ressources** :")
#             for resource in chat['resources']:
#                 st.markdown(f"- {resource}")