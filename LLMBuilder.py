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
    system_prompt: Optional[str] = """You are a chemistry assistant that only answers user's queries about chemistry"""
    context_window: int = 3900
    num_output: int = 512
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
        resp = requests.post(f"{os.getenv('API_BASE_URL')}@cf/meta/llama-3-8b-instruct", headers=headers, json=input_data, stream=True)

        response = ""
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                # Decode the chunk
                decoded_chunk = chunk.decode('utf-8')
                print(decoded_chunk)

                # Find and extract the JSON part (assuming it's within 'data: {...}')
                if decoded_chunk.startswith("data: "):
                    json_string = decoded_chunk.strip()[6:]
                    try:
                        json_data = json.loads(json_string)
                        response_text = json_data.get("response", "")
                        response += response_text
                        yield CompletionResponse(text=response, delta=response_text)
                    except json.JSONDecodeError:
                        return "there was error on the server side"
                    
llm = OurLLM()                    