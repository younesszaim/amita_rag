import streamlit as st
import time
import base64
from streamlit_extras.add_vertical_space import add_vertical_space
st.set_page_config(layout="wide", page_icon='pages/logo/img.png', initial_sidebar_state="collapsed")
st.logo(image='pages/logo/logo.png',size="large")
# --- Load CSS ---
def load_css_file(css_file_path):
    with open(css_file_path) as f:
        return st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css_file('pages/main.css')
# Read the .pptx file in binary mode


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title('Essentiel')
st.subheader("L’organisation interne d’Amita, incluant la présentation institutionnelle, et les structures internes.")
add_vertical_space(2)
# Input field for user question
question = st.text_input("**Wellcome to AmitaGPT! Comment puis-je t'aider ? 😊**", "")

# When the "Réponse" button is clicked
if st.button("Réponse"):
    pass
    if question.strip():
        # Add user's question to chat history
        st.session_state.chat_history.append({"role": "user", "message": question})

        # Display a progress bar
        with st.spinner('Génération de la réponse...'):
            progress_bar = st.progress(0)

            # Simulate response generation process
            for i in range(10):
                time.sleep(0.1)  # Simulate time taken to generate response
                progress_bar.progress((i + 1) * 10)

            # Generate answer using the RAG system
            result = st.session_state.rag.run_rag_prompt(question=question)
            col1, col2 = st.columns([2,1])
            with col1:
                answer = result["response"]
                st.markdown(
                    f"""
                    <div style="text-align: left;">
                        <h3 style="margin-top: 10px;">Réponse</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(answer)
                if "je ne peux pas" or "désolé"  or "je ne comprends pas" in answer:
                    result["resources"] = []
            with col2:
                resources = result["resources"]
                st.markdown(
                    f"""
                    <div style="text-align: left;">
                        <h3 style="margin-top: 10px;">Resources</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                for i in range(len(resources)):
                    #st.write("-", resources[i])
                    with open(st.session_state.data_path + '/'+ resources[i], "rb") as f:
                        data = f.read()
                    # Encode the file to base64
                    b64 = base64.b64encode(data).decode()

                    # Create the download link
                    href = f'<a href="data:application/pdf;base64,{b64}" download="{resources[i]}">{resources[i]}</a>'

                    # Display the clickable file name as a link
                    st.markdown("* " + href, unsafe_allow_html=True)

            # Add assistant's answer to chat history
            st.session_state.chat_history.append(
                {"role": "assistant", "message": answer, "resources": resources})

# Display chat history in reverse order (latest first)
add_vertical_space(10)
st.write("### History Conversation :")
pass
for chat in reversed(st.session_state.chat_history):
    if chat["role"] == "user":
        st.markdown(f"**Vous** : {chat['message']}")
    else:
        st.markdown(f"**AmitaGPT** : {chat['message']}")
        if "resources" in chat:
            st.markdown("**Ressources** :")
            for resource in chat['resources']:
                st.markdown(f"- {resource}")