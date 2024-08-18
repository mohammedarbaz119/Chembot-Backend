from llama_index.core import PromptTemplate
from typing import Any, Dict, List
from llama_index.core import SummaryIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.query_pipeline.query import QueryPipeline
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.core.prompts import PromptTemplate
from LLMBuilder import llm
from IndexBuilder import index
from dotenv import load_dotenv
import os

load_dotenv()

qa_prompt_tmpl_str="""
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge,
respond to the user query below only if it's related to chemistry. If the query is not related to chemistry, respond with "I only answer questions related to chemistry." Do not add information from the context if it is not chemistry-related. If you don't know the answer to a chemistry question, simply respond with "I don't know." Do not attempt to fabricate an answer.
If context is empty or if there is no context present above then say "I don't have information to answer the query." Don't try to make up an answer when the query is not chemistry-related.
Query: {query_str}
Answer:
"""


# qa_prompt_tmpl_str="""
# Context information is below.
# ---------------------
# {context_str}
# ---------------------
# Given the context information and not prior knowledge,
# respond to the user query below only if it's related to chemistry. If the query is not related to chemistry, respond with "I only answer questions related to chemistry." Do not add information from the context if it is not chemistry-related. If you don't know the answer to a chemistry question, simply respond with "I don't know." Do not attempt to fabricate an answer.
# if context is empty then say I don't have information to answer the query. \
# don't try to make up an answer when query is not chemistry related. \
# Query: {query_str}
# Answer:
# """

# qa_prompt_tmpl_str = """\
# Context information is below.
# ---------------------
# {context_str}
# ---------------------
# Given the context information and not prior knowledge, \
# Given the user query below check if it's related to chemistry subject if it's not then say "I only answer questions related to chemistry" and don't try to add from the context. \
# Answer the query asking about chemistry related topics only. \
# If you don't know the answer, just say you don't know, don't try to make up an answer. \
# Query: {query_str}
# Answer: \
# """

qa_prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str
)


system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Retrieved document: \n\n {context_str}
    \n\n User question: {query_str}
    \n\n Evaluation('yes' or 'no'):"""
DEFAULT_RELEVANCY_PROMPT_TEMPLATE = PromptTemplate(
    template=system
)
# DEFAULT_RELEVANCY_PROMPT_TEMPLATE = PromptTemplate(
# #     template1="""As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.
# # """
# template="""Task: Evaluate Document Relevance

#     Retrieved Document:
#     -------------------
#     {context_str}

#     User Question:
#     --------------
#     {query_str}

#     Evaluation Criteria:
#     -------------------
#     Determine if the document contains keywords or topics related to the user's question.
#     The goal is to identify and filter out clearly irrelevant documents.
#     The evaluation should not be overly stringent; the primary objective is to identify and filter out clearly irrelevant retrievals.

#     Decision:
#     ---------
#     - reply with 'yes' if the document is relevant to the question.
#     - reply with 'no' if the document is not relevant.
#     - don't provide any explanation
#     - you should only respond with 'yes' or 'no'.
#     - document relevant (yes or no): """
# )

DEFAULT_TRANSFORM_QUERY_TEMPLATE = PromptTemplate(
    template="""Your task is to refine a query to ensure it is highly effective for retrieving relevant search results. \n
    Analyze the given input to grasp the core semantic intent or meaning. \n
    Original Query:
    \n ------- \n
    {query_str}
    \n ------- \n
    Your goal is to rephrase or enhance this query to improve its search performance. Ensure the revised query is concise and directly aligned with the intended search objective. \n
    Respond with the optimized query only:"""
)
from llama_index.core import get_response_synthesizer

response_synthesizer = get_response_synthesizer(response_mode="refine",llm=llm)

class CorrectiveRAG():
    def __init__(self, index, tavily_ai_apikey: str,llm) -> None:
        """Init params."""
        self.llm = llm
        self.relevancy_pipeline = QueryPipeline(
            chain=[DEFAULT_RELEVANCY_PROMPT_TEMPLATE, llm]
        )
        self.transform_query_pipeline = QueryPipeline(
            chain=[DEFAULT_TRANSFORM_QUERY_TEMPLATE, llm]
        )
        self.index = index
        self.tavily_tool = TavilyToolSpec(api_key=tavily_ai_apikey)

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"llm": self.llm, "index": self.index}

    def retrieve_nodes(self, query_str: str, **kwargs: Any) -> List[NodeWithScore]:
        """Retrieve the relevant nodes for the query."""
        retriever = self.index.as_retriever(**kwargs)
        a = retriever.retrieve(query_str)
        return a

    def evaluate_relevancy(
        self, retrieved_nodes: List[Document], query_str: str
    ) -> List[str]:
        """Evaluate relevancy of retrieved documents with the query."""
        relevancy_results = []
        for node in retrieved_nodes:
            relevancy = self.relevancy_pipeline.run(
                context_str=node.text, query_str=query_str
            )
            relevancy_results.append(relevancy.text)
        return relevancy_results

    def extract_relevant_texts(
        self, retrieved_nodes: List[NodeWithScore], relevancy_results: List[str]
    ) -> str:
        """Extract relevant texts from retrieved documents."""
        relevant_texts = [
            retrieved_nodes[i].text
            for i, result in enumerate(relevancy_results)
            if "yes" in result or "Yes" in result or "YES" in result
        ]
        return "\n".join(relevant_texts)

    def search_with_transformed_query(self, query_str: str) -> str:
        """Search the transformed query with Tavily API."""
        search_results = self.tavily_tool.search(query_str, max_results=5)
        return "\n".join([result.text for result in search_results])

    def get_result(self, relevant_text: str, search_text: str, query_str: str) -> Any:
        """Get result with relevant text."""
        documents = [Document(text=relevant_text + "\n" + search_text)]
        if len(documents[0].text) <=2:
            return "I don't have information to answer the query."
        # index = VectorStoreIndex.from_documents(documents, optimize_for='speed')
        index = SummaryIndex.from_documents(documents)
        query_engine = index.as_query_engine(streaming=True,llm=llm,text_qa_template=qa_prompt_tmpl,similarity_top_k=5)
        return query_engine.query(query_str)
        # response_synthesizer=response_synthesizer,

    def run(self, query_str: str, **kwargs: Any) -> Any:
        """Run the pipeline."""
        # Retrieve nodes based on the input query string.
        retrieved_nodes = self.retrieve_nodes(query_str, **kwargs)

        # Evaluate the relevancy of each retrieved document in relation to the query string.
        relevancy_results = self.evaluate_relevancy(retrieved_nodes, query_str)

        # Extract texts from documents that are deemed relevant based on the evaluation.
        relevant_text = self.extract_relevant_texts(retrieved_nodes, relevancy_results)

        # Initialize search_text variable to handle cases where it might not get defined.
        search_text = ""

        # If any document is found irrelevant, transform the query string for better search results.
        for c in relevancy_results:
             if "yes" not in c.lower():
                 print("no relevant in docs")
                 transformed_query_str = self.transform_query_pipeline.run(
                     query_str=query_str
                   ).text
                 print(transformed_query_str)
            # Conduct a search with the transformed query string and collect the results.
                 search_text = self.search_with_transformed_query(transformed_query_str)
                 break


        # Compile the final result. If there's additional search text from the transformed query,
        # it's included; otherwise, only the relevant text from the initial retrieval is returned.
        if search_text:
            return self.get_result(relevant_text, search_text, query_str)
        else:
            return self.get_result(relevant_text, "", query_str)
crag = CorrectiveRAG(index,os.getenv("TAVILY_API_KEY"),llm)



