import os
import dotenv
import pypdf
import PyPDF2
from PyPDF2 import PdfReader
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
dotenv.load_dotenv()


### SHARE VARIABLES BETWEEN RERUNS --------------------
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'index' not in st.session_state:
    st.session_state['index'] = ''
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = ''
# ------------------------------------------------------------


### forget ========================
def forget():
    st.session_state['messages'] = []
# ====================================


# API key and init embedding
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
EMBEDDER = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def load_document(file):
    docs = []
    reader = PdfReader(file)
    i = 1
    for page in reader.pages:
        docs.append(Document(page_content=page.extract_text(), metadata={'page':i}))
        i += 1

    text_splitter = CharacterTextSplitter(chunk_size=1000, 
                                         chunk_overlap=30, separator="\n")

    return text_splitter.split_documents(documents=docs)


def create_embedding(doc, doc_name, embedder):
    # create and save embeddings
    vectorstore = FAISS.from_documents(doc, embedder)
    vectorstore.save_local(f"document_indexes/{doc_name}")


def load_embedding(doc_name, embedder):
    # load embeddings and return vectorstore to be used as the retriever
    persisted_vectorstore = FAISS.load_local(f"document_indexes/{doc_name}", embedder, allow_dangerous_deserialization=True)
    return persisted_vectorstore


def query_pdf(query, retriever):
   qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY),
                                    chain_type="stuff", retriever=retriever)
   result = qa.run(query)
   return result


def get_available_docs():
    return os.listdir('document_indexes')


def main():
    with st.sidebar:
        st.title('DocAssistant')

        st.markdown('## New Document')
        file = st.file_uploader(label='')
        applied = st.button(label='📄 Load Document', )

        # if apply button clicked
        if applied:

            # if no file uploaded, warn the user
            if not file:
                st.warning('Please load a PDF document first.')

            # if a file is uplaoded, read and create the embedding
            else:
                with st.spinner():
                    doc = load_document(file=file)
                    
                    doc_name = file.name[:file.name.index('.')]

                    if doc_name.strip() in get_available_docs():
                        st.warning('The document is already exists.')
                        
                    else:
                        create_embedding(doc, doc_name, EMBEDDER)
                        st.success('Document load successfully ✔️')


        st.markdown('## Available Documents')
        # list loaded documents
        index = st.selectbox(
            '',
            get_available_docs(),
            on_change=forget,
        )
        st.info(f'{index} is chosen, ready to be queried.')
        st.session_state['index'] = index


        if st.session_state['index'] != '':
            st.session_state['vectorstore'] = load_embedding(index, EMBEDDER)
        
        # st.info(st.session_state['index'])
        # st.info(st.session_state['vectorstore'])
        # st.info(st.session_state['messages'])

        
        add_vertical_space(5)
        st.write('Made by [Mehmet Can DEMIR](<https://www.portfolio-mcandemir.vercel.app/>)')

    
    
    
    ### INPUT BAR ----------------------------------------
    # def chat_input():
    st.chat_input(
        placeholder='Ask your question..',
        key='chat_input',
    )
    # ----------------------------------------------------

    ### SEND MESSAGE ------------------------------------
    # --------------------------------------------------
    if st.session_state['chat_input']:
        st.session_state['messages'].append({
            'role': 'user', 
            'content': st.session_state['chat_input']
            })

        result = query_pdf(
            st.session_state['chat_input'], 
            retriever=st.session_state['vectorstore'].as_retriever()
            )

        st.session_state['messages'].append({'role': 'assistant', 'content': result})


        if len(st.session_state['messages']) > 10:
                st.session_state['messages'].pop(0)
                st.session_state['messages'].pop(0)
        
        print(st.session_state['messages'])
    # --------------------------------------------------



    ### PRINTOUT CHAT HISTORY ----------------------------------------
    # def chat_history():
    if st.session_state['messages']:
        # st.chat_message(name='system').markdown(st.session_state['system_role'][0]['content'])
        for msg in st.session_state['messages']:
            st.chat_message(name=msg['role'], avatar=f'icons/{msg["role"]}.png').markdown(msg['content'])
            # message(msg)
    # ------------------------------------------------------------ 






if __name__ == "__main__":
   main()



























#    filename = input("Enter the name of the document (.pdf or .txt):\n")
   
#    # laod the doc
#    docs = load_document(filename)
   
#    # init embedder
#    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

#    # create and save embeddings
#    vectorstore = FAISS.from_documents(docs, embeddings)
#    vectorstore.save_local("document_indexes/faiss_index_constitution")
   
#    # load embeddings
#    persisted_vectorstore = FAISS.load_local("document_indexes/faiss_index_constitution", embeddings, allow_dangerous_deserialization=True)
   
   
#    query = input("Type in your query (type 'exit' to quit):\n")

#    while query != "exit":
#        query_pdf(query, persisted_vectorstore.as_retriever())
#        query = input("Type in your query (type 'exit' to quit):\n")