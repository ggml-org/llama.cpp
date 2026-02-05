import unittest
from LlamaCppApi import LlamaCppApi

class TestLlamaCppApi(unittest.TestCase):
    def setUp(self):
        # Initializes the LlamaCppApi client for integration tests
        self.client = LlamaCppApi(base_url="http://localhost:8080", api_key="optional_api_key")

    def test_post_completion(self):
        # Test the post_completion method for a successful status code.
        response = self.client.post_completion("Start of a story", {"temperature": 0.9, "n_predict": 15})
        self.assertEqual(response.status_code, 200)
        print(response.json())

    def test_tokenization(self):
        # Test the tokenization endpoint for a successful status code.
        response = self.client.post_tokenize("Example text bob alice eve", {"option_key": "option_value"})
        self.assertEqual(response.status_code, 200)
        print(response.json())


    def test_detokenization(self):
        # Test the detokenization endpoint for a successful status code.
        response = self.client.post_detokenize([13617, 1495, 36292, 71533, 49996], {"option_key": "option_value"})
        self.assertEqual(response.status_code, 200)
        print(response.json())


    def test_health_check(self):
        # Tests the health check endpoint for a successful status code.
        response = self.client.get_health()
        self.assertEqual(response.status_code, 200)
        print(response.json())


    def test_stream_response(self):

        def print_chunk(chunk):
            print("Received Chunk:", chunk)

        response = self.client.stream_response(
            endpoint='completion',
            data={"prompt": "Stream this story", "stream": True, "temperature": 0.7, "n_predict": 32, "stop":["<|im_end|>","<|eot_id|>"]},
            chunk_callback=print_chunk
        )

        print(response)



if __name__ == '__main__':
    unittest.main()
