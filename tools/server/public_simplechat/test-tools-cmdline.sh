echo "DONT FORGET TO RUN llama-server"
echo "build/bin/llama-server -m ~/Downloads/GenAi.Text/gemma-3n-E4B-it-Q8_0.gguf --path tools/server/public_simplechat --jinja"
echo "Note: Remove stream: true line below, if you want one shot instead of streaming response from ai server"
echo "Note: Using different locations below, as the mechanism / url used to fetch will / may need to change"
echo "Note: sudo tcpdump -i lo -s 0 -vvv -A host 127.0.0.1 and port 8080 | tee /tmp/td.log can be used to capture the hs"
curl http://localhost:8080/v1/chat/completions -d '{
    "model": "gpt-3.5-turbo",
    "stream": true,
    "tools": [
      {
        "type":"function",
        "function":{
          "name":"javascript",
          "description":"Runs code in an javascript interpreter and returns the result of the execution after 60 seconds.",
          "parameters":{
            "type":"object",
            "properties":{
              "code":{
                "type":"string",
                "description":"The code to run in the javascript interpreter."
              }
            },
            "required":["code"]
          }
        }
      },
      {
        "type":"function",
        "function":{
          "name":"web_fetch",
          "description":"Connects to the internet and fetches the specified url, may take few seconds",
          "parameters":{
            "type":"object",
            "properties":{
              "url":{
                "type":"string",
                "description":"The url to fetch from internet."
              }
            },
            "required":["url"]
          }
        }
      },
      {
        "type":"function",
        "function":{
          "name":"simple_calc",
          "description":"Calculates the provided arithmatic expression using javascript interpreter and returns the result of the execution after few seconds.",
          "parameters":{
            "type":"object",
            "properties":{
              "arithexp":{
                "type":"string",
                "description":"The arithmatic expression that will be calculated using javascript interpreter."
              }
            },
            "required":["arithexp"]
          }
        }
      }
    ],
    "messages": [
        {
        "role": "user",
        "content": "What and all tools you have access to"
        }
    ]
}'


exit


        "content": "what is your name."
        "content": "What and all tools you have access to"
        "content": "do you have access to any tools"
        "content": "Print a hello world message with python."
        "content": "Print a hello world message with javascript."
        "content": "Calculate the sum of 5 and 27."
        "content": "Can you get me todays date."
        "content": "Can you get me a summary of latest news from bbc world"
        "content": "Can you get todays date. And inturn add 10 to todays date"
        "content": "Who is known as father of the nation in India, also is there a similar figure for USA as well as UK"
        "content": "Who is known as father of the nation in India, Add 10 to double his year of birth and show me the results."
        "content": "How is the weather today in london."
        "content": "How is the weather today in london. Add 324 to todays temperature in celcius in london"
        "content": "How is the weather today in bengaluru. Add 324 to todays temperature in celcius in kochi"
        "content": "Add 324 to todays temperature in celcius in london"
        "content": "Add 324 to todays temperature in celcius in delhi"
        "content": "Add 324 to todays temperature in celcius in delhi. Dont forget to get todays weather info about delhi so that the temperature is valid"
        "content": "Add 324 to todays temperature in celcius in bengaluru. Dont forget to get todays weather info about bengaluru so that the temperature is valid. Use a free weather info site which doesnt require any api keys to get the info"
        "content": "Can you get the cutoff rank for all the deemed medical universities in India for UGNeet 25"
