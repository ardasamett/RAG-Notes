## Corrective Rag with Gemini 1.5 Flash + SKLearnVectorStore + TavilySearchResults + Langgraph + Langsmith
# Do not forget to read my Corrective Rag Medium Post: https://medium.com/@sametarda.dev/deep-dive-into-corrective-rag-implementations-and-workflows-111c0c10b6cf

import os
import uuid
from typing import List
from typing_extensions import TypedDict
from IPython.display import Image, display

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.schema import Document
from langgraph.graph import START, END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults

def set_environment_variables():
    print("Please enter your API keys:")
    
    keys = [
        "GOOGLE_API_KEY",
        "TAVILY_API_KEY",
        "LANGCHAIN_API_KEY"
    ]
    
    for key in keys:
        value = input(f"Enter your {key}: ")
        os.environ[key] = value
    
    # Set other non-sensitive environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "default"
    
    print("Environment variables have been set.")

def initialize_llm():
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)
    model_tested = "gemini-1.5-flash"
    metadata = f"CRAG, {model_tested}"
    return llm, metadata

def load_documents(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    return [item for sublist in docs for item in sublist]

def initialize_vectorstore(docs_list):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=hf_embeddings,
    )
    return vectorstore.as_retriever(k=4)

class RetrievalGrader:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""You are a teacher grading a quiz. You will be given: 
            1/ a QUESTION
            2/ A FACT provided by the student
            
            You are grading RELEVANCE RECALL:
            A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
            A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION. 
            1 is the highest (best) score. 0 is the lowest score you can give. 
            
            Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 
            
            Avoid simply stating the correct answer at the outset.
            
            Question: {question} \n
            Fact: \n\n {documents} \n\n
            
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
            """,
            input_variables=["question", "documents"],
        )
        self.chain = self.prompt | self.llm | JsonOutputParser()
    
    def grade(self, question, documents):
        return self.chain.invoke({"question": question, "documents": documents})

class RAGChain:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. 
            
            Use the following documents to answer the question. 
            
            If you don't know the answer, just say that you don't know. 
            
            Use three sentences maximum and keep the answer concise:
            Question: {question} 
            Documents: {documents} 
            Answer: 
            """,
            input_variables=["question", "documents"],
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def generate(self, question, documents):
        return self.chain.invoke({"documents": documents, "question": question})

class GraphState(TypedDict):
    question: str
    generation: str
    search: str
    documents: List[str]
    steps: List[str]

def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    steps = state["steps"]
    steps.append("retrieve_documents")
    return {"documents": documents, "question": question, "steps": steps}

def generate(state):
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.generate(question, documents)
    steps = state["steps"]
    steps.append("generate_answer")
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "steps": steps,
    }

def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("grade_document_retrieval")
    filtered_docs = []
    search = "No"
    for d in documents:
        score = retrieval_grader.grade(question, d.page_content)
        grade = score["score"]
        if grade == "yes":
            filtered_docs.append(d)
        else:
            search = "Yes"
            continue
    return {
        "documents": filtered_docs,
        "question": question,
        "search": search,
        "steps": steps,
    }

def web_search(state):
    question = state["question"]
    documents = state.get("documents", [])
    steps = state["steps"]
    steps.append("web_search")
    web_results = web_search_tool.invoke({"query": question})
    documents.extend(
        [
            Document(page_content=d["content"], metadata={"url": d["url"]})
            for d in web_results
        ]
    )
    return {"documents": documents, "question": question, "steps": steps}

def decide_to_generate(state):
    search = state["search"]
    if search == "Yes":
        return "search"
    else:
        return "generate"

def setup_workflow():
    workflow = StateGraph(GraphState)
    
    # Define the nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("web_search", web_search)
    
    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def predict_custom_agent_local_answer(example: dict, custom_graph):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state_dict = custom_graph.invoke(
        {"question": example["input"], "steps": []}, config
    )
    return {"response": state_dict["generation"], "steps": state_dict["steps"]}

# Main execution
if __name__ == "__main__":
    set_environment_variables()
    llm, metadata = initialize_llm()
    
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    docs_list = load_documents(urls)
    retriever = initialize_vectorstore(docs_list)
    
    retrieval_grader = RetrievalGrader(llm)
    rag_chain = RAGChain(llm)
    
    web_search_tool = TavilySearchResults(k=3)
    
    custom_graph = setup_workflow()
    
    # Optionally display the graph
    # display(Image(custom_graph.get_graph(xray=True).draw_mermaid_png()))
    
    example = {"input": "What are the types of agent memory?"}
    response = predict_custom_agent_local_answer(example, custom_graph)
    print(response)