openssl req -new -x509 -days 365 -noenc -out /tmp/test-cert.crt -keyout /tmp/test-priv.key -subj '/C=IN/ST=TEST/O=AnveshikaSallap/OU=SimpleMCPTEST/CN=127.0.0.1' -addext "subjectAltName = DNS:localhost, IP:127.0.0.1"
openssl x509 -in /tmp/test-cert.crt -text -noout
#openssl s_client -connect 127.0.0.1:3128 -showcerts
