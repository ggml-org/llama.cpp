echo "DONT FORGET TO RUN simplemcp.py with auth always disabled and in http mode"
echo "Note: sudo tcpdump -i lo -s 0 -vvv -A host 127.0.0.1 and port 3128 | tee /tmp/td.log can be used to capture the hs"
curl http://localhost:3128/mcp --trace - --header "Content-Type: application/json" -d '{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list"
}'

exit

