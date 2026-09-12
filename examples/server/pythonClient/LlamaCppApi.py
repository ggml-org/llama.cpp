import requests
import json

class LlamaCppApi:
    """
    LlamaCppApi provides a Pythonic interface to interact with a server offering 
    various Natural Language Processing (NLP) endpoints, including text generation, 
    tokenization, detokenization, embedding, and server health checks.
    
    :param base_url: The base URL of the NLP server API.
    :param api_key: An optional API key for authentication with the server.
    """
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

    def _send_request(self, method: str, endpoint: str, data: dict = None, params: dict = None, stream: bool = False):
        """
        Sends an HTTP request to the specified endpoint and handles the response, 
        including streaming responses.
        
        :param method: The HTTP method to use ('get' or 'post').
        :param endpoint: The API endpoint to send the request to.
        :param data: The JSON payload for 'post' requests.
        :param params: The query parameters for 'get' requests.
        :param stream: Whether to stream the response.
        :return: The JSON-decoded response data, or None on failure.
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.request(method, url, headers=self.headers, json=data, params=params, stream=stream)
            response.raise_for_status()
            
            if stream:
                return response.iter_lines(decode_unicode=True)
            else:
                return response
        except requests.RequestException as e:
            print(f"Request to {url} failed: {e}")
            return None

    def post_completion(self, prompt: str, options: dict = {}):
        """
        Requests text completion from the server.

        :param prompt: The input text to generate completions for.
        :param options: Additional options for controlling generation.
        :return: Server's response as a JSON object.
        """
        return self._send_request('post', 'completion', data={"prompt": prompt, **options})

    def post_tokenize(self, content: str, options: dict = {}):
        """
        Requests tokenization of the provided content.

        :param content: The text content to tokenize.
        :param options: Additional options for the tokenization request.
        :return: Tokenized content as a JSON object.
        """
        return self._send_request('post', 'tokenize', data={"content": content, **options})

    def post_detokenize(self, tokens: list, options: dict = {}):
        """
        Requests detokenization of the provided tokens.

        :param tokens: The list of tokens to detokenize.
        :param options: Additional options for the detokenization request.
        :return: Detokenized text as a JSON object.
        """
        return self._send_request('post', 'detokenize', data={"tokens": tokens, **options})

    def post_embedding(self, content: str, options: dict = {}):
        """
        Requests embeddings for the provided content.

        :param content: The text content to generate embeddings for.
        :param options: Additional options for the embedding request.
        :return: Embedding data as a JSON object.
        """
        return self._send_request('post', 'embedding', data={"content": content, **options})

    def get_health(self, options: dict = {}):
        """
        Checks the health of the server.

        :param options: Additional options for the health check request.
        :return: Health status as a JSON object.
        """
        return self._send_request('get', 'health', params=options)

    def stream_response(self, endpoint: str, data: dict = {}, chunk_callback = None):
        """
        Handles streaming responses for endpoints that support it, invoking the provided
        callback function for each received chunk of data.
        
        :param endpoint: The API endpoint to send the streaming request to.
        :param data: The request data for streaming endpoints.
        :param chunk_callback: The callback function invoked with each received chunk.
        """
        response_stream = self._send_request('post', endpoint, data=data, stream=True)
        if response_stream:
            for line in response_stream:
                if line.startswith("data: "):
                    try:
                        json_data = json.loads(line.split("data: ", 1)[1])
                        if callable(chunk_callback):
                            chunk_callback(json_data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from streaming response: {e}")
        return response_stream

# Example usage of the LlamaCppApi client
if __name__ == "__main__":
    client = LlamaCppApi(base_url="https://localhost:8080", api_key="YourAPIKey")

    # Requesting text completion with specific options
    prompt = "The meaning of life is"
    options = {"temperature": 0.5, "max_tokens": 50}
    completion_response = client.post_completion(prompt, options=options)
    print("Completion response:", completion_response)

