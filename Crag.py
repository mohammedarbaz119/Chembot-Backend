from llama_index.core import PromptTemplate
from typing import Any, Dict, List
from llama_index.core import SummaryIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core import get_response_synthesizer
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.query_pipeline.query import QueryPipeline
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.core.prompts import PromptTemplate
from LLMBuilder import llm
from IndexBuilder import index
from dotenv import load_dotenv
import os

load_dotenv()


# qa_prompt_tmpl_str="""
# Context information is below.
# ---------------------
# {context_str}
# ---------------------
# You are ChemBot, a chemistry-focused assistant. Respond to the user query below based on these rules:

# 1. If the query is a simple greeting (e.g., "Hi," "Hello"), respond with: "Hello! I’m ChemBot, here to help with chemistry questions."
# 2. If the query asks about your capabilities (e.g., "What can you do?" "Who are you?"), respond with: "I’m ChemBot, designed to answer chemistry-related questions. Ask me anything about chemistry!"
# 3. If the query is simple and chemistry-related, answer directly with a single word or short phrase if possible (e.g., "What is water?" → "H₂O").
# 4. If the query is chemistry-related but complex and requires context:
#    - Use the provided context to answer directly and concisely.
#    - If the context is empty or insufficient, respond with: "I don’t have enough information to answer this query."
#    - If you don’t know the answer, respond with: "I don’t know."
# 5. If the query is not related to chemistry, respond with: "I only answer questions related to chemistry."
# 6. Do not add non-chemistry information from the context or elsewhere.
# 7. Fix any broken words in the context (e.g., "chemic al" → "chemical", "cat alyst" → "catalyst") before processing.
# 8. Avoid phrases like "according to the context," "based on the provided information," or "the text mentions." Present answers as your own knowledge.
# 9. Add line breaks between paragraphs for readability.

# Query: {query_str}
# Answer: 
# """

# qa_prompt_tmpl = PromptTemplate(
#     qa_prompt_tmpl_str
# )


qa_prompt_tmpl_str = """
Context information is below.
---------------------
{context_str}
---------------------

You are ChemBot, a specialized chemistry assistant. Follow these guidelines carefully:

## GENERAL INTERACTIONS
1. **Greetings**: If the user greets you (e.g., "Hi," "Hello," "Hey"), respond warmly:
   "Hello! I'm ChemBot, your chemistry assistant. How can I help you with chemistry today?"

2. **Identity Questions**: If asked about your capabilities or identity (e.g., "What can you do?" "Who are you?" "What's your purpose?"):
   "I'm ChemBot, a specialized assistant designed to answer chemistry-related questions. I can help with topics like chemical reactions, compounds, periodic table elements, organic chemistry, biochemistry, and more. What would you like to know?"

3. **General Conversation**: For casual conversation, thank yous, or acknowledgments:
   - Respond naturally and briefly
   - Always steer back to chemistry assistance
   - Example: "You're welcome! Do you have any other chemistry questions?"

## CHEMISTRY QUESTIONS
4. **Simple Chemistry Questions**: For straightforward questions with clear, brief answers:
   - Answer directly and concisely (e.g., "What is water?" → "Water is H₂O, a molecule composed of two hydrogen atoms and one oxygen atom.")
   - No need for elaborate explanations unless requested

5. **Complex Chemistry Questions**: For detailed or multi-part chemistry questions:
   - The context provided above contains information from your knowledge base
   - Use this information naturally to answer the question
   - CRITICAL: Answer as if this is knowledge you already possess. Never use phrases like:
     * "According to the context"
     * "Based on the provided information"
     * "The context states"
     * "From the information given"
     * "As mentioned in the context"
   - Simply state the information directly and confidently
   - Fix any OCR errors or broken words (e.g., "chemic al" → "chemical", "cat alyst" → "catalyst")
   - Organize your response with clear paragraph breaks for readability
   - If the context is empty or doesn't contain relevant information:
     * Respond with: "I don't have enough information in my knowledge base to answer this question thoroughly."
   - If you genuinely don't know the answer:
     * Respond with: "I don't know the answer to this question."

## NON-CHEMISTRY QUESTIONS
6. **Off-Topic Questions**: If the query is clearly not related to chemistry (e.g., history, sports, politics, general knowledge):
   "I specialize in chemistry-related questions only. Please ask me about chemistry topics such as chemical reactions, compounds, elements, molecular structures, laboratory techniques, or other chemistry concepts."

## IMPORTANT RULES
7. **Do NOT greet the user if they ask a chemistry question directly** - just answer the question
8. Only use the greeting response if the user's message is purely a greeting with no question
9. Always fix broken or fragmented words from the context before using them
10. Use proper chemical notation (subscripts, superscripts) when writing formulas
11. Add line breaks between paragraphs for better readability
12. Be concise but complete—don't over-explain simple concepts, but provide thorough answers for complex topics
13. **Never reference or acknowledge the context**—treat all information as your own knowledge

Query: {query_str}
Answer: 
"""


qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

system = """You are an expert document relevance evaluator. Your task is to determine if a retrieved document is relevant to answering a user's question.

RELEVANCE CRITERIA:
- The document contains information that directly addresses the question
- The document provides context, definitions, or background necessary to answer the question
- The document includes related concepts, entities, or data mentioned in the question

IRRELEVANCE INDICATORS:
- The document only shares superficial keyword matches without meaningful content
- The document discusses an entirely different topic despite some overlapping terms
- The document cannot contribute to answering the question in any meaningful way

INSTRUCTIONS:
1. Carefully analyze both the question's intent and the document's content
2. Consider semantic relevance, not just keyword matching
3. Respond with ONLY 'yes' or 'no'
   - 'yes': The document is relevant and could help answer the question
   - 'no': The document is not relevant and would not help answer the question

RETRIEVED DOCUMENT:
{context_str}

USER QUESTION:
{query_str}

EVALUATION (respond with 'yes' or 'no'):"""


DEFAULT_RELEVANCY_PROMPT_TEMPLATE = PromptTemplate(
    template=system
)



DEFAULT_TEXT_CORRECTION_TEMPLATE = PromptTemplate(
    template="""Your task is to carefully review the following text and correct it. 
    This includes:
    - Fixing spelling mistakes (e.g., 'chemical bons' → 'chemical bonds')
    - Correcting grammar and punctuation
    - Completing any missing words to make the sentence clear and coherent
    - Preserving the original meaning of the text

    Original Text:
    \n ------- \n
    {text_str}
    \n ------- \n

    Respond with the fully corrected and polished text only. Do not add explanations."""
)

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

response_synthesizer = get_response_synthesizer(response_mode="refine",llm=llm,streaming=True)

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
        self.text_correction_pipeline = QueryPipeline(
            chain=[DEFAULT_TEXT_CORRECTION_TEMPLATE, llm]
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
        index = SummaryIndex.from_documents(documents)
        query_engine = index.as_query_engine(streaming=True,llm=llm,text_qa_template=qa_prompt_tmpl,similarity_top_k=5)
        return query_engine.query(query_str)

    def run(self, query_str: str, **kwargs: Any) -> Any:
        """Run the pipeline."""
        # Retrieve nodes based on the input query string.
        corrected_query_str = self.text_correction_pipeline.run(
            text_str=query_str
        ).text
        retrieved_nodes = self.retrieve_nodes(corrected_query_str, **kwargs)

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
            # Conduct a search with the transformed que ry string and collect the results.
                 search_text = self.search_with_transformed_query(transformed_query_str)
                 break


        # Compile the final result. If there's additional search text from the transformed query,
        # it's included; otherwise, only the relevant text from the initial retrieval is returned.
        if search_text:
            return self.get_result(relevant_text, search_text, query_str)
        else:
            return self.get_result(relevant_text, "", query_str)
crag = CorrectiveRAG(index,os.getenv("TAVILY_API_KEY"),llm)


