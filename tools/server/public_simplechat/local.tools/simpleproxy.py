# A simple proxy server
# by Humans for All
#
# Listens on the specified port (defaults to squids 3128)
# * if a url query is got (http://localhost:3128/?url=http://site.of.interest/path/of/interest)
#   fetches the contents of the specified url and returns the same to the requester
#


import sys
import http.server
import urllib.parse
import urllib.request


gMe = {
    '--port': 3128,
    'server': None
}


class ProxyHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        print(f"DBUG:ProxyHandler:{self.path}")
        pr = urllib.parse.urlparse(self.path)
        print(f"DBUG:ProxyHandler:{pr}")
        match pr.path:
            case '/urlraw':
                handle_urlraw(self, pr)
            case '/urltext':
                handle_urltext(self, pr)
            case _:
                print(f"WARN:ProxyHandler:UnknownPath{pr.path}")
                self.send_error(400, f"WARN:UnknownPath:{pr.path}")


def handle_urlraw(ph: ProxyHandler, pr: urllib.parse.ParseResult):
    print(f"DBUG:HandleUrlRaw:{pr}")
    queryParams = urllib.parse.parse_qs(pr.query)
    url = queryParams['url']
    print(f"DBUG:HandleUrlRaw:Url:{url}")
    url = url[0]
    if (not url) or (len(url) == 0):
        ph.send_error(400, "WARN:UrlRaw:MissingUrl")
        return
    try:
        # Get requested url
        with urllib.request.urlopen(url, timeout=10) as response:
            contentData = response.read()
            statusCode = response.status or 200
            contentType = response.getheader('Content-Type') or 'text/html'
        # Send back to client
        ph.send_response(statusCode)
        ph.send_header('Content-Type', contentType)
        # Add CORS for browser fetch, just inc ase
        ph.send_header('Access-Control-Allow-Origin', '*')
        ph.end_headers()
        ph.wfile.write(contentData)
    except Exception as exc:
        ph.send_error(502, f"WARN:UrlFetchFailed:{exc}")


def handle_urltext(ph: ProxyHandler, pr: urllib.parse.ParseResult):
    print(f"DBUG:HandleUrlText:{pr}")
    ph.send_error(400, "WARN:UrlText:Not implemented")


def process_args(args: list[str]):
    global gMe
    gMe['INTERNAL.ProcessArgs.Malformed'] = []
    gMe['INTERNAL.ProcessArgs.Unknown'] = []
    iArg = 1
    while iArg < len(args):
        cArg = args[iArg]
        if (not cArg.startswith("--")):
            gMe['INTERNAL.ProcessArgs.Malformed'].append(cArg)
            print(f"WARN:ProcessArgs:{iArg}:IgnoringMalformedCommandOr???:{cArg}")
            iArg += 1
            continue
        match cArg:
            case '--port':
                iArg += 1
                gMe[cArg] = int(args[iArg])
                iArg += 1
            case _:
                gMe['INTERNAL.ProcessArgs.Unknown'].append(cArg)
                print(f"WARN:ProcessArgs:{iArg}:IgnoringUnknownCommand:{cArg}")
                iArg += 1


def run():
    try:
        gMe['server'] = http.server.HTTPServer(('',gMe['--port']), ProxyHandler)
        gMe['server'].serve_forever()
    except KeyboardInterrupt:
        print("INFO:Run:Shuting down...")
        if (gMe['server']):
            gMe['server'].server_close()
        sys.exit(0)
    except Exception as exc:
        print(f"ERRR:Run:Exception:{exc}")
        if (gMe['server']):
            gMe['server'].server_close()
        sys.exit(1)


if __name__ == "__main__":
    process_args(sys.argv)
    run()
