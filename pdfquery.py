import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFium2Loader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

class PDFQuery:
    def __init__(self, openai_api_key=None):
        """
        Initializes the PDFQuery object.

        Args:
        openai_api_key (str): The API key for OpenAI.
        """
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.chain = None
        self.db = None

    def ask(self, question):
        """
        Asks a question from the ingested document.

        Args:
        question (str): The question to be asked.

        Returns:
        str: The response to the question.
        """
        if self.chain is None:
            return "Please, add a document."
        
        docs = self.db.get_relevant_documents(question)
        return self.chain.run(input_documents=docs, question=question)

    def ingest(self, file_path):
        """
        Ingests a document from a file path.

        Args:
        file_path (str): The path to the file.
        """
        loader = PyPDFium2Loader(file_path)
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)
        self.db = Chroma.from_documents(splitted_documents, self.embeddings).as_retriever()
        self.chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

    def forget(self):
        """
        Removes the ingested documents and resets the chain.
        """
        self.db = None
        self.chain = None
