# Usage:
#! ./llama-server -m some-model.gguf &
#! pip install pydantic
#! python json_schema_pydantic_example.py

from pydantic import BaseModel, Field, TypeAdapter
from annotated_types import MinLen
from typing import Annotated, List, Optional
import json, requests

if True:

    def create_completion(*, response_model=None, endpoint="http://localhost:8080/v1/chat/completions", messages, **kwargs):
        '''
        Creates a chat completion using an OpenAI-compatible endpoint w/ JSON schema support
        (llama.cpp server, llama-cpp-python, Anyscale / Together...)

        The response_model param takes a type (+ supports Pydantic) and behaves just as w/ Instructor (see below)
        '''
        response_format = None
        type_adapter = None

        if response_model:
            type_adapter = TypeAdapter(response_model)
            schema = type_adapter.json_schema()
            messages = [{
                "role": "system",
                "content": f"You respond in JSON format with the following schema: {json.dumps(schema, indent=2)}"
            }] + messages
            response_format={"type": "json_object", "schema": schema}

        data = requests.post(endpoint, headers={"Content-Type": "application/json"},
                             json=dict(messages=messages, response_format=response_format, **kwargs)).json()
        if 'error' in data:
            raise Exception(data['error']['message'])

        content = data["choices"][0]["message"]["content"]
        return type_adapter.validate_json(content) if type_adapter else content

else:

    # This alternative branch uses Instructor + OpenAI client lib.
    # Instructor support streamed iterable responses, retry & more.
    # (see https://python.useinstructor.com/)
    #! pip install instructor openai
    import instructor, openai
    client = instructor.patch(
        openai.OpenAI(api_key="123", base_url="http://localhost:8080"),
        mode=instructor.Mode.JSON_SCHEMA)
    create_completion = client.chat.completions.create


if __name__ == '__main__':

    class QAPair(BaseModel):
        class Config:
            extra = 'forbid'  # triggers additionalProperties: false in the JSON schema
        question: str
        concise_answer: str
        justification: str
        stars: Annotated[int, Field(ge=1, le=5)]

    class PyramidalSummary(BaseModel):
        class Config:
            extra = 'forbid'  # triggers additionalProperties: false in the JSON schema
        title: str
        summary: str
        question_answers: Annotated[List[QAPair], MinLen(2)]
        sub_sections: Optional[Annotated[List['PyramidalSummary'], MinLen(2)]]

    print("# Summary\n", create_completion(
        model="...",
        response_model=PyramidalSummary,
        messages=[{
            "role": "user",
            "content": f"""
                You are a highly efficient corporate document summarizer.
                Create a pyramidal summary of an imaginary internal document about our company processes
                (starting high-level, going down to each sub sections).
                Keep questions short, and answers even shorter (trivia / quizz style).
            """
        }]))
