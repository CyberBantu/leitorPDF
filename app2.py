import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv
import os 
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('hkunlp/instructor-xl')


with st.sidebar:
    st.title('App Feito com LLCHAIN')

    add_vertical_space(10)
    st.write('Feito por Christian Basilio [Linkedin](Linkedin.com)')




def main():
    st.write('Converse com o relatório!')

    load_dotenv()

    # Fazendo upload do arquivo
    pdf = st.file_uploader('Faça o Upload do PDF', type = 'pdf')

    # pra evitar o erro
    #st.write(pdf)
    if pdf is not None:

        st.write(pdf.name)

        leitor_de_pdf = PdfReader(pdf)

        text = ''

        # Mostra o texto
        for page in leitor_de_pdf.pages:
            text += page.extract_text()
        
        # Leitura do texto com langchain
        text_lang = RecursiveCharacterTextSplitter(
            chunk_size = 5000, # possiveis divições
            chunk_overlap = 200, # Numero de caracteres
            length_function =len # interavel
        )
        chunks = text_lang.split_text(text = text)

        # Embading - Verific conexoes semanticas entre palavras
        embedding = OpenAIEmbeddings()

        # usando o faisss
        VectorStore = FAISS.from_texts(chunks, embedding)

        # Tirando o .pdf do nome
        novo_nome = pdf.name[:-4]

        # Criando arquivo caso nao tenha nada --------------
        if os.path.exists(f"{novo_nome}.pkl"):
            with open(f"{novo_nome}.pkl", 'rb') as f:
                VectorStore = pickle.load(f)
          #  st.wite('Embeding no arquivo')
        else:
            # Embading - Verific conexoes semanticas entre palavras
            embedding = OpenAIEmbeddings()

            # usando o faisss
            VectorStore = FAISS.from_texts(chunks, embedding)

            with open(f"{novo_nome}.pkl", 'wb') as f:
                pickle.dump(VectorStore, f)


        # Aceitação de questoes e perguntas

        query = st.text_input('Pergunte ao PDF:')

        # Escrevendo o que ele colocou
        st.write(query)


        if query:
        # analisando
            docs = VectorStore.similarity_search(query = query, k=3)

            llm = OpenAI(model_name='gpt-3.5-turbo')

            chain = load_qa_chain(llm=llm, chain_type='stuff') # Tipo de leitura


            # Criando call back
            with get_openai_callback() as cb:

                response = chain.run(input_documents = docs, question = query)
                print(cb)

            st.write(response)






        



        #st.write(chunks)


        # Faz aparecer o nome do pdf carregado
        #st.write(text)

if __name__ == '__main__':
    main()