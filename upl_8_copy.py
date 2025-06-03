# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 22:42:04 2023


"""
import warnings
warnings.filterwarnings("ignore")

from bs4 import BeautifulSoup as bs4
import spacy_streamlit
from spacy_streamlit import visualize_ner
from spacy_streamlit import load_model
import streamlit as st
from streamlit_extras.app_logo import add_logo
import spacy
import PyPDF2
import requests
from spacy.matcher import Matcher 
import pandas as pd
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
import os
import json
from custom_scorer_2 import Custom_Scorer

os.chdir(r'C:\Users\Final_Project\Blocks')

# Load SpaCy model
output_dir = r'C:\Users\Final_Project\content'
#print("Loading from", output_dir)
with st.sidebar:
  st.image("SpaCy_logo.svg", width=150)
  st.image("1326689.png", width=250)
spacy_model = st.sidebar.selectbox("Model name", ("en_core_web_lg", "en_core_web_sm", output_dir), index=None, placeholder = "Choose a Model",)

if spacy_model is not None:
    nlp2 = spacy.load(spacy_model)




# Streamlit app
st.title("Machine Learning NER Custom Model and Network Graph Visualization")


# Form for URL input
with st.form(key='url_form'):
    st.write("Enter a URL:")
    url_input = st.text_input(label="", key='url_input')
    submit_button = st.form_submit_button(label='Process URL')
    

# File uploader
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt"])

if uploaded_file is not None:
    # Read the content of the file
    file_type = uploaded_file.name.split('.')[-1]
    st.write(f"File type: {file_type}")

    if file_type == "pdf":
        pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
        # Extract the content
        content = ""
        for page in range(pdf_reader.getNumPages()):
            content += pdf_reader.getPage(page).extractText()
        # Display the content
        st.write(content)
        #content)
        
    elif file_type == "txt":
        content = uploaded_file.getvalue().decode("utf-8")
        st.header("NER of Text File Content:")
        # st.write(content)
else: 
    content = ""
#creating Doc - uploaded from txt or PDF 
if spacy_model is not None:       
    doc = nlp2(content)
# st.write("LENGTH OF DOC: ", len(doc))
if spacy_model is not None:
 if len(doc) !=0:
    visualize_ner(doc, labels=nlp2.get_pipe("ner").labels, key = 1)

#OUTPUT of all entites and its labels
#st.write(([(ent.text, ent.label_) for ent in doc.ents]))
        
if 'url_input' in st.session_state:
    url = st.session_state.url_input
    if submit_button:
    # If a URL is provided, fetch its content using a session
        session = requests.Session()
        try:
            response = session.get(url).content
            # Create BS4 object to handle HTML data
            soup = bs4(response, "lxml")
            # Extract text from body tag and remove \n, \s and \t
            content_1 = soup.find("body").text.strip()
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching URL: {e}")
            content_1 = ""
    else:
        content_1 = ""
else:
    content_1 = ""
    
    
# Process the text with SpaCy
#creating Doc1 - uploaded from URL 
if spacy_model is not None:
    doc1 = nlp2(content_1)
# st.write("LENGTH OF DOC_1: ", len(doc1))

# Display the processed content
if spacy_model is not None:
 if len(doc1) !=0:
    st.subheader("NER of Processed Text Body from URL:")
#st.text(doc.text)

    # Display named entities
    #st.write("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        
    visualize_ner(doc1, labels=nlp2.get_pipe("ner").labels, key = 2)
#st.write(([(ent.text, ent.label_) for ent in doc1.ents]))
#st.write(doc)

#splitting test text into sentences
Lorem_ipsum = """Lorem ipsum dolor sit amet,
consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua."""
if spacy_model is not None:
 if len(doc1) !=0:
    sentences_1 = [sent.text.strip() for sent in doc1.sents]
else: sentences_1 = Lorem_ipsum


if spacy_model is not None:
 if len(doc) !=0:
    sentences = [sent.text.strip() for sent in doc.sents]
else: sentences = Lorem_ipsum

#checks of Docs and Sents of NLP2
# st.write("THIS IS DOC_1", doc1) 
# st.write("THIS IS DOC", doc)    
# st.write("THIS IS SENTENCES", sentences)
# st.write("THIS IS SENTENCES_1", sentences_1)

#Get Relations for the entities
def get_relation(sent):

  doc = nlp2(sent)

  # Matcher class object 
  matcher = Matcher(nlp2.vocab)

  #define the pattern 
  pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"}] 

  matcher.add("matching_1",[pattern]) 

  matches = matcher(doc)
  if not matches:
      return ""
  k = len(matches) - 1
  span = doc[matches[k][1]:matches[k][2]]

  return(span.text)

# if len(doc) !=0:
#     st.write("THIS IS THE FIRST SENTENCE EDGE FROM DOC: ", get_relation(sentences[0]))
# else: st.write("DOC IS NONE: ", Lorem_ipsum)

# if len(doc) !=0:
#     st.write(sentences)

# if len(doc1) !=0:
#     st.write("THIS IS THE FIRST SENTENCE EDGE FROM DOC_1: ", get_relation(sentences_1[0]))
# else: st.write("DOC_1 IS NONE: ", Lorem_ipsum)

# Count the number of entities in each sentence
entity_counts_per_sentence = []
if spacy_model is not None:
 if len(doc) !=0:
    for sent in sentences:
    # Create a SpaCy Doc object for each sentence
        sent_doc = nlp2(sent)
    
    # Count the number of entities in the sentence
        entity_count = len(sent_doc.ents)
    
    # Get the names of entities in the sentence
        entity_names = [ent.text for ent in sent_doc.ents]
    
    # Store the result
        entity_counts_per_sentence.append((sent, entity_count, entity_names))

# Print the results
# for sent, count, names in entity_counts_per_sentence:
#     print(f"Sentence: {sent}")
#     print(f"Number of entities: {count}")
#     print(f"Entity names: {names}")
#     print(get_relation(sent))
#     print("------")
    
# Create a DataFrame

entity_counts_per_sentence = []
if spacy_model is not None:
 if len(doc) !=0:
    for sent in sentences:
        sent_doc = nlp2(sent)
        entity_count = len(sent_doc.ents)
        entity_names = [ent.text for ent in sent_doc.ents]
        relation = get_relation(sent)
        if entity_count >= 2:
            entity_counts_per_sentence.append((sent, entity_names[0], entity_names[1:], relation))




# Create a DataFrame

df = pd.DataFrame(entity_counts_per_sentence, columns=['Sentence', 'Source', 'Targets', 'Relation'])
if df.empty is False:
    st.subheader("Dataframe with source data of graph chart")
    st.write(df)

# st.write("This is a sentences_1", sentences_1)

entity_counts_per_sentence1 = []
if spacy_model is not None:
 if len(doc1) !=0:
    for sent in sentences_1:
        sent_doc1 = nlp2(sent)
        entity_count = len(sent_doc1.ents)
        entity_names1 = [ent.text for ent in sent_doc1.ents]
        relation1 = get_relation(sent)
        if entity_count >= 2:
            entity_counts_per_sentence1.append((sent, entity_names1[0], entity_names1[1:], relation1))

# Create a DataFrame
df1 = pd.DataFrame(entity_counts_per_sentence1, columns=['Sentence', 'Source', 'Targets', 'Relation'])
if df1.empty is False:
    st.subheader("Dataframe with source data of graph chart")
    st.write(df1)

#evaluating Precision-Recall metrics of Custom model
if df.empty is False or df1.empty is False:
    custom_scorer = Custom_Scorer()
    CONTROL_1 = custom_scorer.evaluate()
    
    st.subheader("Precision, Recall and F1 Score of the Model")
    st.write("Precision:", CONTROL_1['ents_p'])
    st.write("Recall:", CONTROL_1['ents_r'])
    st.write("F1 Score:", CONTROL_1['ents_f'])
    st.write("Scores per type:", CONTROL_1['ents_per_type'])
    
    

# Assuming you already have a DataFrame df with columns 'Source', 'Targets', 'Edge'
os.chdir(r'C:\Users\Final_Project\Blocks')

if df.empty is False:
    kg_df = df
elif df1.empty is False:
    kg_df = df1


# Create a directed graph
if df.empty is False or df1.empty is False:
    st.subheader("Network of Named Entities, detected by trained custom model")
    G = Network(notebook=True, directed=True,
    
    bgcolor='#222222',
    font_color='white',
    height='600px',)

    # Add nodes and edges to the graph
    for _, row in kg_df.iterrows():
        source = row['Source']
        targets = row['Targets']
        relation = row['Relation']
    
        # Add source node
        G.add_node(source)
    
        # Add target nodes and edges with labels
        if isinstance(targets, list):
            for target in targets:
                G.add_node(target, color='red')
                G.add_edge(source, target, label=relation, font_color="red")  # Use label for edge labels
    
                
    
     # Save and read graph as HTML file (locally)
    path = 'html_files'
    G.save_graph(f'{path}/graphy88.html')
    HtmlFile = open(f'{path}/graphy88.html', 'r', encoding='utf-8')
         
     # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=935)    
