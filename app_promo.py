import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import CohereEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma, FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, LLMChainExtractor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from htmlTemplates import css, bot_template, user_template

from langchain.retrievers import BM25Retriever, EnsembleRetriever

from langchain.llms import OpenAI

from pdf_ocr import *


def chunk_docs(list_of_pages):
    all_text = ''
    for page in list_of_pages:
        all_text += ' \n'.join(page)

    # all_text = '\n\n'.join(chunks)
    # print(type(all_text))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                  chunk_overlap=100,
                  add_start_index=True,
                  separators = ['\n\n', '\n', " ", ""]
                                              ) # keep all paragraphs (and then sentences, and then words) together as long as possible

    chunked_docs = splitter.split_text(all_text)
    return chunked_docs


def get_text_chunks(text):
    #from V1
    chunks = []
    for i,page in enumerate(text):
        chunks.append(f'Página {int(i + 1)}: \n' + '\n'.join(page))

    return chunks


from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def build_rag_chain(retriever):
##https://js.langchain.com/docs/modules/chains/document/refine
##https://stackoverflow.com/questions/77521745/map-reduce-prompt-with-retrievalqa-chain

    llm = ChatOpenAI(model='gpt-4', temperature=0)

    question_prompt = PromptTemplate(input_variables=['question', 'context_str'],
                                     template='''Eres un asistente de inteligencia artificial experto en pólizas de seguros.
                                     Dada una pregunta de un usuario y un apartado de su póliza de seguros, debes responder a la pregunta del usuario y mencionar la página que contiene la respuesta.
                                     La respuesta a la pregunta puede estar contenida en una tabla,
                                     Mantén la respuesta lo más precisa posible.
                                     Apartado: "{context_str}"
                                     Pregunta: "{question}"
                                     ''')



    refine_prompt = PromptTemplate(input_variables=['question', 'existing_answer', 'context_str'],
                                template='''Eres un asistente de inteligencia artificial experto en pólizas de seguros.
                                Dada una pregunta de un usuario y un apartado de su póliza de seguros, debes responder a la pregunta del usuario y mencionar la página que contiene la respuesta.
                                La pregunta original es: "{question}"
                                Esta es una respuesta generada previamente: "{existing_answer}"
                                * Importante: Si la respuesta encontrada es precisa devuelve unicamente la respuesta encontrada.
                                Existe la posibilidad de que esta respuesta sea más precisa si se agrega informacion del siguiente apartado de la póliza:
                                ---
                                Apartado: "{context_str}"
                                ---
                                Dado el nuevo apartado, complementa o corrige la respuesta encontrada (solo si es preciso y necesario) con este contexto adicional.
                                No ólvides referenciar, en orden, todas las páginas relevantes.
                                El usuario no sabe que estas haciendo este paso, por lo tanto, devuelve ÚNICAMENTE una respuesta a la pregunta original.
                                ''')

    qa_chain = RetrievalQA.from_chain_type(llm = llm,
                                    chain_type = "refine",
                                    retriever = retriever,
                                    chain_type_kwargs={
                                        "question_prompt": question_prompt,
                                        "refine_prompt" : refine_prompt
                                    }
                                           )

    return qa_chain


llm_promo = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def analyze_query(user_question):
    messages = [
        ("system", f'''Eres un asistente de inteligencia artificial experto en analizar consultas de usuario.
                       Dada una pregunta de un usuario, debes responder únicamente:
                        - True: si el usuario pregunta su póliza de seguros del usuario.
                        - False: si la pregunta es una consulta sobre contratar una nueva póliza de seguros.
                        
                       La respuesta a la pregunta debe conter únicamente la palabra "True" o la palabra "False".
                       
        --Ejemplos de preguntas que debes responder con "True":

        De cuanto es mi prima asegurada?
        Cuanto es mi recargo financiero?
        Cuanto es la prima total?
        Cual es la vigencia de la poliza?
        Cual es mi condición de pago?
        ¿Cuál es la cobertura que brinda la póliza de Responsabilidad Civil de Auto?
        ¿Cuales son los gastos de expedición?
        ¿Cuál es la cobertura por robo de piezas o accesorios de la póliza y cuáles son las limitaciones?
        Cual es la cobertura de Gastos medicos ocupantes?
        ¿Cuáles son los límites mínimos de responsabilidad por lesiones corporales y daños a la propiedad requeridos en EE. UU. y Canadá?
        Cuál es valor d la cobertura por muerte del titular?
        Cual es la cobertura de daños a terceros?
        ¿Cuáles son las condiciones para que esté vigente la póliza de Responsabilidad Civil de Autos?
        ¿Qué riesgos y bienes están asegurados por Chubb Seguros México, S.A.?
                       
                       '''),
        ("human", user_question),
    ]
    response = llm_promo.invoke(messages)
    generated_text = response.content

    return "true" in generated_text.lower()


def get_promo_answer(user_question):
    messages = [
        ("system", f'''Eres un asistente de inteligencia artificial experto en promocionales de seguros Afirme.
                       IMPORTANTE: tu respuesta debe escapar (\) caracteres con efectos de formato markdown.
                       Dada una pregunta de un usuario sobre una promoción de seguros de mascotas, debes recomendar que el usuario hable con su agente de seguros Afirme para adquirir la siguiente promoción de seguros de mascotas:
                       
        SEGURO MASCOTA
        ¡Esta es una promoción exclusiva para ti!

        Al contratar tu seguro de auto, podrás adquirir a un costo preferencial un seguro para proteger a tu mascota desde $75 al mes. Te brindamos una suma asegurada de hasta $8,000 (vía reembolso) para proteger y atender a tu mascota.

        Aplica para:

        Gastos médicos por accidente
        Gastos médicos por enfermedad no prevenible
        Indemnización por muerte accidental o enfermedad no prevenible
        Hospedaje de la Mascota en caso de hospitalización del propietario
        Servicios de asistencia incluidos:

        Vacuna antirrábica o desparasitación
        Consulta de valoración
        Asistencia veterinaria telefónica
        Asistencia legal
        Orientación telefónica para el transporte de mascotas vía aérea
        Concierge mascota
        Precio anual: $900

        ¡Aprovecha esta promoción! Adquiérelo en este momento.

        Restricciones aplicables:

        Cubre perros y gatos.
        La cobertura aplica hasta 5 mascotas hasta la suma asegurada de $8,000 pesos (suma asegurada única para la vigencia de la póliza para las 5 mascotas).
        Los servicios de asistencia de una consulta de valoración gratuita y una vacuna antirrábica o una desparasitación gratuita aplican para un evento al año para cualquiera de tus 5 mascotas (1 servicio).
        El servicio de una vacuna antirrábica o una desparasitación gratuita tiene un periodo de espera de 120 días, solo para estas 2 asistencias aplican hasta una edad de la mascota de hasta 5 años. Es requisito contar con cartilla de vacunación actualizada por cada mascota.
        La cobertura de gastos de hospedaje de la mascota aplica solo en caso de hospitalización del propietario.
        Costo anual del Seguro Mascota $900.00, la forma de pago de este seguro es de acuerdo con la frecuencia y forma de pago que confirmes en la compra del Seguro de Auto. Esta oferta no aplica para venta individual y estás sujeto a la contratación del Seguro de Auto.'''),
        ("human", user_question),
    ]
    response = llm_promo.invoke(messages)
    generated_text = response.content

    return generated_text


def handle_userinput(user_question, conversation_chain):
    is_insurance_query = analyze_query(user_question)

    if is_insurance_query:
        response = conversation_chain({'query': user_question})
        answer = response["result"]
    else:
        answer = get_promo_answer(user_question)

    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="RAGente de póliza", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    st.header("RAGente de póliza :books:")
    user_question = st.text_input("Haz una consulta sobre tu póliza:")

    with st.sidebar:
        st.subheader("Carga de documentos")
        pdf_docs = st.file_uploader("Sube tu póliza en PDF aquí y da click en 'Procesar'", accept_multiple_files=True)
        if st.button("Procesar"):
            with st.spinner("Procesando documentos"):
                raw_text = process_pdf(pdf_docs[0], table_model)
                # text_chunks = chunk_docs(raw_text)
                text_chunks = get_text_chunks(raw_text) ## use this for whole page context

                # embeddings = CohereEmbeddings(model="embed-multilingual-light-v3.0")

                # embeddings = OpenAIEmbeddings()

                # retriever = FAISS.from_texts(texts=text_chunks, embedding=embeddings)\
                #                 .as_retriever(search_type="similarity_score_threshold",
                #                             search_kwargs={"score_threshold": 0.65, "k": 3})


                # initialize the bm25 retriever and faiss retriever
                bm25_retriever = BM25Retriever.from_texts(
                    text_chunks
                )
                bm25_retriever.k = 2

                embedding = OpenAIEmbeddings()
                faiss_vectorstore = FAISS.from_texts(
                    text_chunks, embedding
                )
                faiss_retriever = faiss_vectorstore.as_retriever(search_type="similarity_score_threshold",
                                                                search_kwargs={"score_threshold": 0.65, "k": 2})

                # initialize the ensemble retriever
                retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, faiss_retriever], weights=[0.50, 0.50]
                )

                st.session_state.conversation_chain = build_rag_chain(retriever)

    if user_question and st.session_state.conversation_chain:
        handle_userinput(user_question, st.session_state.conversation_chain)


if __name__ == '__main__':
    main()
