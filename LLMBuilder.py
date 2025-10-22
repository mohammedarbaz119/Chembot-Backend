from typing import Optional, List, Mapping, Any
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()



headers = {"Authorization": f"Bearer {os.getenv('cloudfareToken')}"}

class OurLLM(CustomLLM):
    system_prompt: Optional[str] = """You are a chemistry assistant specializing in chemistry-related queries and document grading. Your responses should be clear, accurate, and use proper line breaks between paragraphs.
For general greetings or non-chemistry queries:

For chemistry queries:

Provide detailed, scientifically accurate answers
Include relevant formulas, equations, and explanations when needed
Break complex concepts into understandable parts
Use proper line breaks between paragraphs for readability

When grading documents:

Evaluate content against the provided query
Highlight key chemistry concepts and their accuracy
Point out any misconceptions or errors
Provide constructive feedback
Use clear section breaks between different assessment points

Always maintain scientific accuracy and use appropriate chemistry terminology while keeping explanations accessible."""
    
    context_window: int = 4096
    num_output: int = 1024
    model_name: str = "CloudfareLLMLLama3"
    dummy_response: str = "My response"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        inputs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{prompt}"}
        ]
        input_data = {"messages": inputs}
        response = requests.post(f"{os.getenv('API_BASE_URL')}@cf/meta/llama-3-8b-instruct", headers=headers, json=input_data)

        return CompletionResponse(text=response.json()["result"]["response"])

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        inputs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{prompt}"}
        ]
        input_data = {"messages": inputs, "stream": True}
        try:
            resp = requests.post(f"{os.getenv('API_BASE_URL')}@cf/meta/llama-3-8b-instruct", headers=headers, json=input_data, stream=True)

            response = ""
            for chunk in resp.iter_lines(decode_unicode=True):
                if chunk:
                    # Decode the chunk
                    decoded_chunk = chunk

                    if decoded_chunk.startswith("data: "):
                        json_string = decoded_chunk.strip()[6:]
                        try:
                            json_data = json.loads(json_string)
                            response_text = json_data.get("response", "")
                            response += response_text
                            yield CompletionResponse(text=response, delta=response_text)
                        except json.JSONDecodeError:
                            return "there was error on the server side"
        except Exception as e:
            print(e)
            yield e        
                    
llm = OurLLM()                    
