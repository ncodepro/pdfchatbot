# Import necessary libraries for pdf loading, vectorization, embeddings,
# LLM model, and question answering chain.
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

# Set OpenAI API key as an environment variable.
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"

# Specify the path to the pdf file.
pdf_path = "https://navalmanack.s3.amazonaws.com/Eric-Jorgenson_The-Almanack-of-Naval-Ravikant_Final.pdf"

# Create a PyPDFLoader object with the pdf path.
loader = PyPDFLoader(pdf_path)

# Load and split the pages of the pdf document.
pages = loader.load_and_split()

# Create an OpenAIEmbeddings object for embeddings.
embeddings = OpenAIEmbeddings()

# Create a Chroma object from documents for vectorization, and transform it into a document retriever.
docsearch = Chroma.from_documents(pages, embeddings).as_retriever()

# Define the query for the model.
query = "What does Naval's thought about on how to be happy"

# Retrieve the documents relevant to the query.
docs = docsearch.get_relevant_documents(query)

# Load a question answering chain with OpenAI's model (temperature of 0 means deterministic responses).
# Chain_type is set to "stuff", but this would depend on the actual library's implementation.
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

# Run the question answering chain with the relevant documents and the query.
output = chain.run(input_documents=docs, question=query)

# Print the output from the question answering chain.
print(output)
