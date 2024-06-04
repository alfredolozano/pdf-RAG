import streamlit as st
from dotenv import load_dotenv
# from PyPDF2 import PdfReader
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


import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure the language model for the application.")
    parser.add_argument("--llm", type=str, default="gpt-4", help="Specify the language model to use. Default is 'gpt-4'.")
    args = parser.parse_args()
    return args.llm

#language_model = parse_arguments()
language_model = 'gpt-4'

# st.set_page_config(page_title="Agente conversacional", page_icon="app/static/logo.jpg")
# st.markdown(css, unsafe_allow_html=True)

history_chat = "history_chat"

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

    llm = ChatOpenAI(model=language_model, temperature=0)

    question_prompt = PromptTemplate(input_variables=['question', 'context_str'],
                                     template='''Eres un asistente experto en pólizas de seguros.
                                     Dada una pregunta de un usuario y un apartado de su póliza de seguros, debes responder a la pregunta del usuario y mencionar la página que contiene la respuesta. Utiliza elementos de sintaxis como listas, bullets y salto de línea para optimizar la experiencia de usuario.
                                     La respuesta a la pregunta puede estar contenida en una tabla,
                                     Mantén la respuesta lo más clara y precisa posible para el usuario final.
                                     Incluye teléfono o algún medio de atención si la consulta lo requiere, especialmente para reportar siniestros.
                                     Apartado: "{context_str}"

                                     Pregunta: "{question}"
                                     '''
                                     )



    refine_prompt = PromptTemplate(input_variables=['question', 'existing_answer', 'context_str'],
                                template='''Eres un asistente experto en pólizas de seguros dedicado a proveer respuestas útiles al usuario.
                                Dada una pregunta de un usuario y un apartado de su póliza de seguros, debes responder a la pregunta del usuario y mencionar la página que contiene la respuesta. Utiliza elementos de sintaxis como listas, bullets y salto de línea para optimizar la experiencia de usuario.
                                ---
                                La pregunta original es: "{question}"
                                ---
                                Esta es una respuesta generada previamente: "{existing_answer}"
                                ---

                                * Importante: Si la respuesta previa es precisa devuelve unicamente la respuesta previa sin agregar información.
                                Existe la posibilidad de que esta respuesta sea más precisa si se agrega informacion del siguiente apartado de la póliza:
                                
                                ---
                                Apartado: "{context_str}"
                                ---
                                
                                Instrucciones:
                                - Incluye teléfono o algún medio de atención si la consulta lo requiere, especialmente para reportar siniestros.
                                - No debes agregar información que no esté relacionada a la pregunta o meta comentarios sobre las diferentes respuestas como "La respuesta original...".
                                - Dado el nuevo apartado, complementa o corrige la respuesta previa (solo si es preciso y necesario) con este contexto adicional.
                                - Debes referenciar, en orden, todas las páginas relevantes.
                                - Para pólizas con cobertura en México y el extranjero, prioriza la información válida en México.
                                - El usuario no sabe que estas haciendo este paso de refinamiento, por lo tanto, devuelve ÚNICAMENTE una respuesta a la pregunta original, sin comentarios como "la respuesta original es precisa" ni referencias a distintas respuestas.'''
                                )

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
                        - True: si el usuario pregunta exclusivamente sobre promociones o seguro de mascotas.
                        - False: en cualquier caso contrario.

                       Es muy importate que identifiques cuando el usuario quiere información sobre un nuevo seguro de mascotas, cualquier pregunta que no esté estrictamente relacionada a un seguro de mascotas debe ser respondida como "False".

                       Ejemplos de preguntas con respuesta "True":
                       1. me puedes compartir información sobre un seguro de mascotas?
                       2. me gustaría obtener información sobre un seguro de mascotas
                       3. quiero obtener información sobre un seguro de mascotas

                       La respuesta a la pregunta debe conter únicamente la palabra "True" o la palabra "False".
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

        - Vacuna antirrábica o desparasitación
        - Consulta de valoración
        - Asistencia veterinaria telefónica
        - Asistencia legal
        - Orientación telefónica para el transporte de mascotas vía aérea
        - Concierge mascota
        - Precio anual: $900

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


def write_chat(history):
    for qa in history:
        question = qa["question"]
        answer  = qa["answer"]
        st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

def escape_dollar_signs(text):
    return text.replace('$', '&#36;')

import time
def handle_userinput(user_question, conversation_chain):
    start_time = time.time()

    is_pet_query = analyze_query(user_question)

    if not is_pet_query:
        response = conversation_chain({'query': user_question})
        answer = response["result"]
        if answer[0] == '"' and answer[-1] == '"': answer = answer[1:-1]
    else:
        answer = get_promo_answer(user_question)

    # Escape dollar signs in the answer
    escaped_answer = escape_dollar_signs(answer)
    elapsed_time = time.time() - start_time

    print(f"Escaped Answer: {escaped_answer}")
    print(f"Time to process: {elapsed_time} seconds")

    st.session_state[history_chat].append({"answer": escaped_answer, "question": user_question})
    write_chat(st.session_state[history_chat])

def process_documents():
    if history_chat in st.session_state:
        st.session_state[history_chat] = []

    with st.sidebar:
        if 'pdf_docs' in st.session_state and st.session_state.pdf_docs:
            with st.spinner("Procesando documentos"):
                raw_text = process_pdf(st.session_state.pdf_docs[0], table_model)
                text_chunks = get_text_chunks(raw_text) ## use this for whole page context
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
                retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, faiss_retriever], weights=[0.50, 0.50]
                )
                st.session_state.conversation_chain = build_rag_chain(retriever)
                st.markdown(css, unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Agente conversacional", page_icon="app/static/logo.jpg")
    st.markdown(css, unsafe_allow_html=True)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    if history_chat not in st.session_state:
        st.session_state[history_chat] = []

    st.header("¡Hola! Bienvenido a AFIRME ChatBot, estoy aquí para ayudarte a resolver cualquier duda que tengas sobre alguna de tus pólizas. Pregunta también por nuestras promociones")
    user_question = st.text_input("Haz una consulta sobre tu póliza")

    with st.sidebar:
        st.subheader("Carga de documentos")
        st.file_uploader("Sube tu póliza en PDF aquí", accept_multiple_files=True, on_change=process_documents, key="pdf_docs")


    if user_question and st.session_state.conversation_chain:
        handle_userinput(user_question, st.session_state.conversation_chain)


if __name__ == '__main__':
    main()
