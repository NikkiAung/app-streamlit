import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("Codebasics Q&A 🌱")
btn = st.button("Create Knowledgebase")

if btn:
    create_vector_db()
    st.success("Knowledgebase created successfully!")

question = st.text_input("Enter your question:")
if question:
    chain = get_qa_chain()
    answer = chain.invoke({"query": question})
    st.write("Answer:", answer)
