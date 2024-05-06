import streamlit as st
from time import time
import requests
import PyPDF2
import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import string
stop_words = set(nltk.corpus.stopwords.words('english') + list(string.punctuation))
endpoint = "https://902b-34-171-12-202.ngrok-free.app/summarize"

# Pre-Processing when taking text from pdf
def preprocess_text(text):
    # Remove non-alphanumeric characters except for spaces and common punctuation
    text = re.sub(r'[^A-Za-z0-9, .?!]', '', text)
    return text

def select_files():
    files = st.file_uploader("Choose Files", type=["pdf"], accept_multiple_files=True)
    texts = []
    sentences = []
    for i, file in enumerate(files):
        st.markdown("---")
        t0 = time()
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            t = page.extract_text()
            sentences.extend(sent_tokenize(preprocess_text(t)))
            text += t
        text = preprocess_text(text)
        st.caption("Text from %s, processed in %0.3fs" % (file.name, time() - t0))
        st.markdown(f"<div style='height:{15}em; overflow-y: auto;'>{text}</div>", unsafe_allow_html=True)
        texts.append(text)
    return sentences

def find_topics(sentences):
    if len(sentences) == 0:
        return None
    tokenized_sentences = [' '.join(nltk.word_tokenize(text.lower())) for text in sentences]
    filtered_sentences = [' '.join([word for word in text.split() if word not in stop_words]) for text in tokenized_sentences]

    # Create a bag-of-words representation using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(filtered_sentences)
    # Build the LDA model
    lda_model = LatentDirichletAllocation(n_components=2, random_state=42)
    lda_model.fit(X)
    # Assign topics to each document
    document_topics = lda_model.transform(X)

    # Print topic labels and their scores
    topic_labels = [f"Topic {i + 1}" for i in range(len(lda_model.components_))]
    st.write("## Topic Labels:")
    st.write(", ".join(topic_labels))

    # Group documents based on similar topics
    grouped_documents = {}
    for doc_idx, doc_topics in enumerate(document_topics):
        top_topic_index = doc_topics.argmax()

        if top_topic_index not in grouped_documents:
            grouped_documents[top_topic_index] = []

        grouped_documents[top_topic_index].append(doc_idx)

    return grouped_documents

# Rest of the code remains unchanged


# Rest of the code remains unchanged

    if len(sentences) == 0:
        return None
    tokenized_sentences = [' '.join(nltk.word_tokenize(text.lower())) for text in sentences]
    filtered_sentences = [' '.join([word for word in text.split() if word not in stop_words]) for text in tokenized_sentences]

    # Create a bag-of-words representation using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(filtered_sentences)
    # Build the LDA model
    lda_model = LatentDirichletAllocation(n_components=2, random_state=42)
    lda_model.fit(X)
    # Assign topics to each document
    document_topics = lda_model.transform(X)
    # Group documents based on similar topics
    grouped_documents = {}
    for doc_idx, doc_topics in enumerate(document_topics):
        top_topic_index = doc_topics.argmax()

        if top_topic_index not in grouped_documents:
            grouped_documents[top_topic_index] = []

        grouped_documents[top_topic_index].append(doc_idx)
    return grouped_documents

def heading():
    st.title('Multi-Document Summarization')
    st.caption('Unlocking Insights with Multi-Document Summarization: Harnessing the power of Llama-2 Model for precision summaries and leveraging LDA methods to cluster text groups for a comprehensive understanding.')

def summary(sentences):
    if(len(sentences)==0):
        return
    output = ""
    t0 = time()
    response = requests.post(url=endpoint,data={'text':' '.join(sentences)})
    output += "**"+response.json()['title']+"**\n\n"
    output += response.json()['summary']+"\n\n"
    st.success("Completed in %0.3fs"%(time()-t0))
    with st.container(border=True):
        st.markdown(output)

if __name__ == '__main__':
    with st.sidebar:
        sentences = select_files()
    heading()
    col1,_, col2 = st.columns(3)
    if col1.button("Run LDA"):
        grouped_documents = find_topics(sentences)
        if grouped_documents is not None:
            st.write(f"## LDA Based Grouping")
            with st.container(border=True):
                ind = 0
                for topic, docs in grouped_documents.items():
                    with st.container(border=True):
                        st.write(f"##### Cluster {ind+1}")  
                        text = ""
                        for doc_idx in docs:
                            text += sentences[doc_idx] + " "
                        st.markdown(f"<div style='height:{15}em; overflow-y: auto;'>{text}</div>", unsafe_allow_html=True)
                        st.write("\n")
                      
                    ind += 1
    if col2.button("Summarize"):
        summary(sentences)