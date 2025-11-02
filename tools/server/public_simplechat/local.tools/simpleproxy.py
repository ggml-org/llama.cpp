# A simple proxy server
# by Humans for All
#
# Listens on the specified port (defaults to squids 3128)
# * if a url query is got wrt urlraw path
#   http://localhost:3128/urlraw?url=http://site.of.interest/path/of/interest
#   fetches the contents of the specified url and returns the same to the requester
# * if a url query is got wrt urltext path
#   http://localhost:3128/urltext?url=http://site.of.interest/path/of/interest
#   fetches the contents of the specified url and returns the same to the requester
#   after removing html tags in general as well as contents of tags like style
#   script, header, footer, nav ...
# * any request to aum path is used to respond with a predefined text response
#   which can help identify this server, in a simple way.
#
# Expects a Bearer authorization line in the http header of the requests got.
# HOWEVER DO KEEP IN MIND THAT ITS A VERY INSECURE IMPLEMENTATION, AT BEST
#


import sys
import http.server
import urllib.parse
import urllib.request
from dataclasses import dataclass
import html.parser
import re
import time
import urlvalidator as uv


gMe = {
    '--port': 3128,
    '--config': '/dev/null',
    '--debug': False,
    'bearer.transformed.year': "",
    'server': None
}

gConfigType = {
    '--port': 'int',
    '--config': 'str',
    '--debug': 'bool',
    '--allowed.schemes': 'list',
    '--allowed.domains': 'list',
    '--bearer.insecure': 'str'
}

gConfigNeeded = [ '--allowed.schemes', '--allowed.domains', '--bearer.insecure' ]

gAllowedCalls = [ "urltext", "urlraw", "pdf2text" ]


def bearer_transform():
    """
    Transform the raw bearer token to the network handshaked token,
    if and when needed.
    """
    global gMe
    year = str(time.gmtime().tm_year)
    if gMe['bearer.transformed.year'] == year:
        return
    import hashlib
    s256 = hashlib.sha256(year.encode('utf-8'))
    s256.update(gMe['--bearer.insecure'].encode('utf-8'))
    gMe['--bearer.transformed'] = s256.hexdigest()
    gMe['bearer.transformed.year'] = year


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    """
    Implements the logic for handling requests sent to this server.
    """

    def send_headers_common(self):
        """
        Common headers to include in responses from this server
        """
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """
        Overrides the SendError helper
        so that the common headers mentioned above can get added to them
        else CORS failure will be triggered by the browser on fetch from browser.
        """
        print(f"WARN:PH:SendError:{code}:{message}")
        self.send_response(code, message)
        self.send_headers_common()

    def auth_check(self):
        """
        Simple Bearer authorization
        ALERT: For multiple reasons, this is a very insecure implementation.
        """
        bearer_transform()
        authline = self.headers['Authorization']
        if authline == None:
            return { 'AllOk': False, 'Msg': "No auth line" }
        authlineA = authline.strip().split(' ')
        if len(authlineA) != 2:
            return { 'AllOk': False, 'Msg': "Invalid auth line" }
        if authlineA[0] != 'Bearer':
            return { 'AllOk': False, 'Msg': "Invalid auth type" }
        if authlineA[1] != gMe['--bearer.transformed']:
            return { 'AllOk': False, 'Msg': "Invalid auth" }
        return { 'AllOk': True, 'Msg': "Auth Ok" }

    def do_GET(self):
        """
        Handle GET requests
        """
        print(f"\n\n\nDBUG:ProxyHandler:GET:{self.address_string()}:{self.path}")
        print(f"DBUG:PH:Get:Headers:{self.headers}")
        pr = urllib.parse.urlparse(self.path)
        print(f"DBUG:ProxyHandler:GET:{pr}")
        match pr.path:
            case '/urlraw':
                acGot = self.auth_check()
                if not acGot['AllOk']:
                    self.send_error(400, f"WARN:{acGot['Msg']}")
                else:
                    handle_urlraw(self, pr)
            case '/urltext':
                acGot = self.auth_check()
                if not acGot['AllOk']:
                    self.send_error(400, f"WARN:{acGot['Msg']}")
                else:
                    handle_urltext(self, pr)
            case '/pdf2text':
                acGot = self.auth_check()
                if not acGot['AllOk']:
                    self.send_error(400, f"WARN:{acGot['Msg']}")
                else:
                    handle_pdf2text(self, pr)
            case '/aum':
                handle_aum(self, pr)
            case _:
                print(f"WARN:ProxyHandler:GET:UnknownPath{pr.path}")
                self.send_error(400, f"WARN:UnknownPath:{pr.path}")

    def do_OPTIONS(self):
        """
        Handle OPTIONS for CORS preflights (just in case from browser)
        """
        print(f"DBUG:ProxyHandler:OPTIONS:{self.path}")
        self.send_response(200)
        self.send_headers_common()


def handle_aum(ph: ProxyHandler, pr: urllib.parse.ParseResult):
    """
    Handle requests to aum path, which is used in a simple way to
    verify that one is communicating with this proxy server
    """
    queryParams = urllib.parse.parse_qs(pr.query)
    url = queryParams['url']
    print(f"DBUG:HandleAUM:Url:{url}")
    url = url[0]
    if (not url) or (len(url) == 0):
        ph.send_error(400, f"WARN:HandleAUM:MissingUrl/UnknownQuery?!")
        return
    urlParts = url.split('.',1)
    if not (urlParts[0] in gAllowedCalls):
        ph.send_error(403, f"WARN:HandleAUM:Forbidded:{urlParts[0]}")
        return
    print(f"INFO:HandleAUM:Availability ok for:{urlParts[0]}")
    ph.send_response_only(200, "bharatavarshe")
    ph.send_header('Access-Control-Allow-Origin', '*')
    ph.end_headers()


@dataclass(frozen=True)
class UrlReqResp:
    """
    Used to return result wrt urlreq helper below.
    """
    callOk: bool
    httpStatus: int
    httpStatusMsg: str = ""
    contentType: str = ""
    contentData: str = ""


def debug_dump(meta: dict, data: dict):
    if not gMe['--debug']:
        return
    timeTag = f"{time.time():0.12f}"
    with open(f"/tmp/simpleproxy.{timeTag}.meta", '+w') as f:
        for k in meta:
            f.write(f"\n\n\n\n{k}:{meta[k]}\n\n\n\n")
    with open(f"/tmp/simpleproxy.{timeTag}.data", '+w') as f:
        for k in data:
            f.write(f"\n\n\n\n{k}:{data[k]}\n\n\n\n")


def handle_urlreq(ph: ProxyHandler, pr: urllib.parse.ParseResult, tag: str):
    """
    Common part of the url request handling used by both urlraw and urltext.

    Verify the url being requested is allowed.

    Include User-Agent, Accept-Language and Accept in the generated request using
    equivalent values got in the request being proxied, so as to try mimic the
    real client, whose request we are proxying. In case a header is missing in the
    got request, fallback to using some possibly ok enough defaults.

    Fetch the requested url.
    """
    tag=f"UrlReq:{tag}"
    queryParams = urllib.parse.parse_qs(pr.query)
    url = queryParams['url']
    print(f"DBUG:{tag}:Url:{url}")
    url = url[0]
    gotVU = uv.validate_url(url, tag)
    if not gotVU.callOk:
        return UrlReqResp(gotVU.callOk, gotVU.statusCode, gotVU.statusMsg)
    try:
        hUA = ph.headers.get('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0')
        hAL = ph.headers.get('Accept-Language', "en-US,en;q=0.9")
        hA = ph.headers.get('Accept', "text/html,*/*")
        headers = {
            'User-Agent': hUA,
            'Accept': hA,
            'Accept-Language': hAL
        }
        req = urllib.request.Request(url, headers=headers)
        # Get requested url
        print(f"DBUG:{tag}:Req:{req.full_url}:{req.headers}")
        with urllib.request.urlopen(req, timeout=10) as response:
            contentData = response.read().decode('utf-8')
            statusCode = response.status or 200
            contentType = response.getheader('Content-Type') or 'text/html'
            debug_dump({ 'url': req.full_url, 'headers': req.headers, 'ctype': contentType }, { 'cdata': contentData })
        return UrlReqResp(True, statusCode, "", contentType, contentData)
    except Exception as exc:
        return UrlReqResp(False, 502, f"WARN:{tag}:Failed:{exc}")


def handle_urlraw(ph: ProxyHandler, pr: urllib.parse.ParseResult):
    try:
        # Get requested url
        got = handle_urlreq(ph, pr, "HandleUrlRaw")
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
        ph.send_error(502, f"WARN:UrlRawFailed:{exc}")


class TextHtmlParser(html.parser.HTMLParser):
    """
    A simple minded logic used to strip html content of
    * all the html tags as well as
    * all the contents belonging to below predefined tags like script, style, header, ...

    NOTE: if the html content/page uses any javascript for client side manipulation/generation of
    html content, that logic wont be triggered, so also such client side dynamic content wont be
    got.

    This helps return a relatively clean textual representation of the html file/content being parsed.
    """

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
        """
        Helps decide whether to capture contents or discard them.
        """
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
        got = handle_urlreq(ph, pr, "HandleUrlText")
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
        debug_dump({ 'RawText': 'yes', 'StrippedText': 'yes' }, { 'RawText': textHtml.text, 'StrippedText': textHtml.get_stripped_text() })
    except Exception as exc:
        ph.send_error(502, f"WARN:UrlTextFailed:{exc}")


def process_pdf2text(url: str, startPN: int, endPN: int):
    import pypdf
    import io
    gotVU = uv.validate_url(url, "HandlePdf2Text")
    if not gotVU.callOk:
        return { 'status': gotVU.statusCode, 'msg': gotVU.statusMsg }
    urlParts = urllib.parse.urlparse(url)
    fPdf = open(urlParts.path, 'rb')
    dPdf = fPdf.read()
    tPdf = ""
    oPdf = pypdf.PdfReader(io.BytesIO(dPdf))
    if (startPN < 0):
        startPN = 0
    if (endPN < 0) or (endPN >= len(oPdf.pages)):
        endPN = len(oPdf.pages)-1
    for i in range(startPN, endPN+1):
        pd = oPdf.pages[i]
        tPdf = tPdf + pd.extract_text()
    return { 'status': 200, 'msg': "Pdf2Text Response follows", 'data': tPdf }


def handle_pdf2text(ph: ProxyHandler, pr: urllib.parse.ParseResult):
    """
    Handle requests to pdf2text path, which is used to extract plain text
    from the specified pdf file.
    """
    queryParams = urllib.parse.parse_qs(pr.query)
    url = queryParams['url'][0]
    startP = queryParams['startPageNumber'][0]
    if startP:
        startP = int(startP)
    else:
        startP = -1
    endP = queryParams['endPageNumber'][0]
    if endP:
        endP = int(endP)
    else:
        endP = -1
    print(f"INFO:HandlePdf2Text:Processing:{url}:{startP}:{endP}...")
    gotP2T = process_pdf2text(url, startP, endP)
    if (gotP2T['status'] != 200):
        ph.send_error(gotP2T['status'], gotP2T['msg'] )
        return
    ph.send_response(gotP2T['status'], gotP2T['msg'])
    ph.send_header('Content-Type', 'text/text')
    # Add CORS for browser fetch, just in case
    ph.send_header('Access-Control-Allow-Origin', '*')
    ph.end_headers()
    print(f"INFO:HandlePdf2Text:ExtractedText:{url}...")
    ph.wfile.write(gotP2T['data'].encode('utf-8'))



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
            print(f"DBUG:LoadConfig:{k}")
            try:
                cArg = f"--{k}"
                aTypeCheck = gConfigType[cArg]
                aValue = cfg[k]
                aType = type(aValue).__name__
                if aType != aTypeCheck:
                    print(f"ERRR:LoadConfig:{k}:expected type [{aTypeCheck}] got type [{aType}]")
                    exit(112)
                gMe[cArg] = aValue
            except KeyError:
                print(f"ERRR:LoadConfig:{k}:UnknownCommand")
                exit(113)


def process_args(args: list[str]):
    """
    Helper to process command line arguments.

    Flow setup below such that
    * location of --config in commandline will decide whether command line or config file will get
    priority wrt setting program parameters.
    * str type values in cmdline are picked up directly, without running them through ast.literal_eval,
    bcas otherwise one will have to ensure throught the cmdline arg mechanism that string quote is
    retained for literal_eval
    """
    import ast
    global gMe
    iArg = 1
    while iArg < len(args):
        cArg = args[iArg]
        if (not cArg.startswith("--")):
            print(f"ERRR:ProcessArgs:{iArg}:{cArg}:MalformedCommandOr???")
            exit(101)
        print(f"DBUG:ProcessArgs:{iArg}:{cArg}")
        try:
            aTypeCheck = gConfigType[cArg]
            aValue = args[iArg+1]
            if aTypeCheck != 'str':
                aValue = ast.literal_eval(aValue)
                aType = type(aValue).__name__
                if aType != aTypeCheck:
                    print(f"ERRR:ProcessArgs:{iArg}:{cArg}:expected type [{aTypeCheck}] got type [{aType}]")
                    exit(102)
            gMe[cArg] = aValue
            iArg += 2
            if cArg == '--config':
                load_config()
        except KeyError:
            print(f"ERRR:ProcessArgs:{iArg}:{cArg}:UnknownCommand")
            exit(103)
    print(gMe)
    for k in gConfigNeeded:
        if gMe.get(k) == None:
            print(f"ERRR:ProcessArgs:{k}:missing, did you forget to pass the config file...")
            exit(104)
    uv.validator_setup(gMe['--allowed.schemes'], gMe['--allowed.domains'])


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
