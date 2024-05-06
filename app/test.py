import streamlit as st
import requests
#Flask API
endpoint = "https://9149-34-71-48-41.ngrok-free.app/summarize"
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#For Clustering
from collections import defaultdict

#Text Splitter Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)

def lda_analysis(texts, num_topics=1):
    # Vectorize the text data
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Display the top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords_idx = topic.argsort()[:-10-1:-1]
        top_keywords = [feature_names[i] for i in top_keywords_idx]
    return top_keywords

def cluster_sentence(sentence_labels):
    # Set the threshold for common labels
    threshold = 2 # Adjust as needed

    # Create a dictionary to store clusters
    clusters = defaultdict(list)

    assigned_sentences = set()

    # Iterate through each pair of sentences
    for sentence1, labels1 in sentence_labels.items():
        # Skip if the sentence is already assigned to a cluster
        if sentence1 in assigned_sentences:
            continue
        
        # Create a new cluster for the current sentence
        current_cluster = [sentence1]
        
        # Mark the current sentence as assigned
        assigned_sentences.add(sentence1)

        for sentence2, labels2 in sentence_labels.items():
            # Skip if the sentence is already assigned to a cluster
            if sentence2 in assigned_sentences:
                continue

            # Count the number of common labels
            common_labels = len(set(labels1).intersection(labels2))

            # Check if the number of common labels exceeds the threshold
            if common_labels >= threshold:
                # Append sentence to the current cluster
                current_cluster.append(sentence2)
                
                # Mark the current sentence as assigned
                assigned_sentences.add(sentence2)

        # Append the current cluster to the list of clusters
        clusters[sentence1] = current_cluster

    # Convert the clusters to a list of clusters
    result_clusters = list(clusters.values())
    return result_clusters

def main():
    st.title("LDA Topic Modeling with Streamlit")

    # Get user input
    text_input = st.text_area("Enter text:", "Your text goes here.")

    if st.button("Run LDA"):
        output = ""
        t0 = time()
        sentences = text_splitter.split_text(text_input)
        sentence_labels = {}
        for sentence in sentences:
            sentence_labels[sentence] = lda_analysis([sentence])
        clusters = cluster_sentence(sentence_labels=sentence_labels)
        progress_bar = st.progress(0)
        for i,cluster in enumerate(clusters):
            response = requests.post(url=endpoint,data={'text':' '.join(cluster)})
            output += "**"+response.json()['title']+"**\n\n"
            output += response.json()['summary']+"\n\n"
            progress_bar.progress((i + 1) / len(clusters))
        progress_bar.empty()
        st.success("Completed in %0.3fs"%(time()-t0))
        st.markdown(output)

if __name__ == "__main__":
    main()
