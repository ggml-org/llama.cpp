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
from dataclasses import dataclass
import html.parser


gMe = {
    '--port': 3128,
    '--config': '/dev/null',
    'server': None
}


class ProxyHandler(http.server.BaseHTTPRequestHandler):

    # Common headers to include in responses from this server
    def send_headers_common(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    # overrides the SendError helper
    # so that the common headers mentioned above can get added to them
    # else CORS failure will be triggered by the browser on fetch from browser.
    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        self.send_response(code, message)
        self.send_headers_common()

    # Handle GET requests
    def do_GET(self):
        print(f"DBUG:ProxyHandler:GET:{self.path}")
        pr = urllib.parse.urlparse(self.path)
        print(f"DBUG:ProxyHandler:GET:{pr}")
        match pr.path:
            case '/urlraw':
                handle_urlraw(self, pr)
            case '/urltext':
                handle_urltext(self, pr)
            case '/aum':
                handle_aum(self, pr)
            case _:
                print(f"WARN:ProxyHandler:GET:UnknownPath{pr.path}")
                self.send_error(400, f"WARN:UnknownPath:{pr.path}")

    # Handle OPTIONS for CORS preflights (just in case from browser)
    def do_OPTIONS(self):
        print(f"DBUG:ProxyHandler:OPTIONS:{self.path}")
        self.send_response(200)
        self.send_headers_common()


def handle_aum(ph: ProxyHandler, pr: urllib.parse.ParseResult):
    ph.send_response_only(200, "bharatavarshe")
    ph.send_header('Access-Control-Allow-Origin', '*')
    ph.end_headers()


@dataclass(frozen=True)
class UrlReqResp:
    callOk: bool
    httpStatus: int
    httpStatusMsg: str = ""
    contentType: str = ""
    contentData: str = ""


def handle_urlreq(pr: urllib.parse.ParseResult, tag: str):
    print(f"DBUG:{tag}:{pr}")
    queryParams = urllib.parse.parse_qs(pr.query)
    url = queryParams['url']
    print(f"DBUG:{tag}:Url:{url}")
    url = url[0]
    if (not url) or (len(url) == 0):
        return UrlReqResp(False, 400, f"WARN:{tag}:MissingUrl")
    try:
        # Get requested url
        with urllib.request.urlopen(url, timeout=10) as response:
            contentData = response.read().decode('utf-8')
            statusCode = response.status or 200
            contentType = response.getheader('Content-Type') or 'text/html'
        return UrlReqResp(True, statusCode, "", contentType, contentData)
    except Exception as exc:
        return UrlReqResp(False, 502, f"WARN:UrlFetchFailed:{exc}")


def handle_urlraw(ph: ProxyHandler, pr: urllib.parse.ParseResult):
    try:
        # Get requested url
        got = handle_urlreq(pr, "HandleUrlRaw")
        if not got.callOk:
            ph.send_error(got.httpStatus, got.httpStatusMsg)
            return
        # Send back to client
        ph.send_response(got.httpStatus)
        ph.send_header('Content-Type', got.contentType)
        # Add CORS for browser fetch, just in case
        ph.send_header('Access-Control-Allow-Origin', '*')
        ph.end_headers()
        ph.wfile.write(got.contentData.encode('utf-8'))
    except Exception as exc:
        ph.send_error(502, f"WARN:UrlFetchFailed:{exc}")


class TextHtmlParser(html.parser.HTMLParser):

    def __init__(self):
        super().__init__()
        self.inside = {
            'body': False,
            'script': False,
            'style': False,
            'header': False,
            'footer': False,
            'nav': False
        }
        self.monitored = [ 'body', 'script', 'style', 'header', 'footer', 'nav' ]
        self.bCapture = False
        self.text = ""
        self.textStripped = ""

    def do_capture(self):
        if self.inside['body'] and not (self.inside['script'] or self.inside['style'] or self.inside['header'] or self.inside['footer'] or self.inside['nav']):
            return True
        return False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag in self.monitored:
            self.inside[tag] = True

    def handle_endtag(self, tag: str):
        if tag in self.monitored:
            self.inside[tag] = False

    def handle_data(self, data: str):
        if self.do_capture():
            self.text += f"{data}\n"

    def syncup(self):
        self.textStripped = self.text

    def strip_adjacent_newlines(self):
        oldLen = -99
        newLen = len(self.textStripped)
        aStripped = self.textStripped;
        while oldLen != newLen:
            oldLen = newLen
            aStripped = aStripped.replace("\n\n\n","\n")
            newLen = len(aStripped)
        self.textStripped = aStripped

    def strip_whitespace_lines(self):
        aLines = self.textStripped.splitlines()
        self.textStripped = ""
        for line in aLines:
            if (len(line.strip())==0):
                self.textStripped += "\n"
                continue
            self.textStripped += f"{line}\n"

    def get_stripped_text(self):
        self.syncup()
        self.strip_whitespace_lines()
        self.strip_adjacent_newlines()
        return self.textStripped


def handle_urltext(ph: ProxyHandler, pr: urllib.parse.ParseResult):
    try:
        # Get requested url
        got = handle_urlreq(pr, "HandleUrlText")
        if not got.callOk:
            ph.send_error(got.httpStatus, got.httpStatusMsg)
            return
        # Extract Text
        textHtml = TextHtmlParser()
        textHtml.feed(got.contentData)
        # Send back to client
        ph.send_response(got.httpStatus)
        ph.send_header('Content-Type', got.contentType)
        # Add CORS for browser fetch, just in case
        ph.send_header('Access-Control-Allow-Origin', '*')
        ph.end_headers()
        ph.wfile.write(textHtml.get_stripped_text().encode('utf-8'))
    except Exception as exc:
        ph.send_error(502, f"WARN:UrlFetchFailed:{exc}")


def load_config():
    """
    Allow loading of a json based config file

    The config entries should be named same as their equivalent cmdline argument
    entries but without the -- prefix. They will be loaded into gMe after adding
    -- prefix.

    As far as the program is concerned the entries could either come from cmdline
    or from a json based config file.
    """
    global gMe
    import json
    with open(gMe['--config']) as f:
        cfg = json.load(f)
        for k in cfg:
            gMe[f"--{k}"] = cfg[k]


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
            case '--config':
                iArg += 1
                gMe[cArg] = args[iArg]
                iArg += 1
                load_config()
            case _:
                gMe['INTERNAL.ProcessArgs.Unknown'].append(cArg)
                print(f"WARN:ProcessArgs:{iArg}:IgnoringUnknownCommand:{cArg}")
                iArg += 1
    print(gMe)


def run():
    try:
        gMe['serverAddr'] = ('', gMe['--port'])
        gMe['server'] = http.server.HTTPServer(gMe['serverAddr'], ProxyHandler)
        print(f"INFO:Run:Starting on {gMe['serverAddr']}")
        gMe['server'].serve_forever()
    except KeyboardInterrupt:
        print("INFO:Run:Shuting down...")
        if (gMe['server']):
            gMe['server'].server_close()
        sys.exit(0)
    except Exception as exc:
        print(f"ERRR:Run:Exiting:Exception:{exc}")
        if (gMe['server']):
            gMe['server'].server_close()
        sys.exit(1)


if __name__ == "__main__":
    process_args(sys.argv)
    run()
