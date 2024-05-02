# TAMU-CS-RAG
A Retrieval-Augmented Generation (RAG) pipeline built on Computer Science Department Courses data for empowering Q&amp;A chatbot for answering queries of students. The goal of this project is to explore different techniques for Retrieval
Augmented Generation and build a question answering chat not that can answer question
related to any user specific data. For testing the implementation of the RAG model, we use the
Texas A&M Computer Science courses syllabus data for answering question related to the
syllabus for the courses offered. The essential step in RAG pipeline if how we capture the data
into chunks which can be indexed for efficient retrieval. Breaking unstructured data into
semantically logical chucks which will also have information about the global topic and
document was a major challenge that was tackled in this project. The text chucks were formed
with document title and details like position in metadata which was then transformed into
embedding vectors using Google Gemini model. These embeddings were stored in vector
database for quick retrieval. Finally, when there is a user question the text chunks with high
semantic similarity were fetched, along with chat history and compressed into a prompt with
relevant data as context for the model to answer. The large language model can now answer the
question from the context provided in the prompt. A application was made using Streamlit fin
python for displaying the answers in a chat format for the user.

Example 1:

<img width="865" alt="Screenshot 2024-05-01 at 9 40 51 PM" src="https://github.com/chinmay10/TAMU-CS-RAG/assets/22643060/7258934d-7e5d-48ee-87dd-1e5c56100b30">
<img width="865" alt="Screenshot 2024-05-01 at 9 41 11 PM" src="https://github.com/chinmay10/TAMU-CS-RAG/assets/22643060/c8992dc6-0639-442e-9cd4-19c9fe9bde01">



Example 2:

<img width="865" alt="Screenshot 2024-05-01 at 10 26 42 PM" src="https://github.com/chinmay10/TAMU-CS-RAG/assets/22643060/788b3230-b6fc-409c-b627-3aac2386f9a6">
<img width="865" alt="Screenshot 2024-05-01 at 10 27 01 PM" src="https://github.com/chinmay10/TAMU-CS-RAG/assets/22643060/f32ff831-39cb-49f3-948e-c3f73a4eba5e">
