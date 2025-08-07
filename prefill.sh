echo "Show false settings" >> query.txt
cat mypyini query.txt >full.txt
# precompute mypy.ini into prefill.save
rm prefill.save
./llama-cli -m ~/funstreams/granite-3.1-1b-a400m-instruct-IQ4_NL.gguf -c 2000 -st --prompt-cache prefill.save -f mypy.ini -n 1

# use prefill.save to answer query
./llama-cli -m /funstreams/granite-3.1-1b-a400m-instruct-IQ4_NL.gguf -c 2000 -st --prompt-cache prefill.save -f full.txt --prompt-cache-ro

