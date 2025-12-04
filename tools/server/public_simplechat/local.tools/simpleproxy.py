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
import time
import ssl
import traceback
from typing import Callable
import urlvalidator as uv
import pdfmagic as mPdf
import webmagic as mWeb
import debug as mDebug


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
    '--bearer.insecure': 'str',
    '--sec.keyfile': 'str',
    '--sec.certfile': 'str'
}

gConfigNeeded = [ '--allowed.schemes', '--allowed.domains', '--bearer.insecure' ]

gAllowedCalls = {
    "xmlfiltered": [],
    "htmltext": [],
    "urlraw": [],
    "pdftext": [ "pypdf" ]
    }


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

    def auth_and_run(self, pr:urllib.parse.ParseResult, handler:Callable[['ProxyHandler', urllib.parse.ParseResult], None]):
        """
        If authorisation is ok for the request, run the specified handler.
        """
        acGot = self.auth_check()
        if not acGot['AllOk']:
            self.send_error(400, f"WARN:{acGot['Msg']}")
        else:
            try:
                handler(self, pr)
            except Exception as e:
                self.send_error(400, f"ERRR:ProxyHandler:{e}")

    def do_GET(self):
        """
        Handle GET requests
        """
        print(f"DBUG:ProxyHandler:GET:{self.address_string()}:{self.path}")
        print(f"DBUG:PH:Get:Headers:{self.headers}")
        pr = urllib.parse.urlparse(self.path)
        print(f"DBUG:ProxyHandler:GET:{pr}")
        match pr.path:
            case '/urlraw':
                self.auth_and_run(pr, mWeb.handle_urlraw)
            case '/htmltext':
                self.auth_and_run(pr, mWeb.handle_htmltext)
            case '/xmlfiltered':
                self.auth_and_run(pr, mWeb.handle_xmlfiltered)
            case '/pdftext':
                self.auth_and_run(pr, mPdf.handle_pdftext)
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

    def handle(self) -> None:
        print(f"\n\n\nDBUG:ProxyHandler:Handle:RequestFrom:{self.client_address}")
        return super().handle()


def handle_aum(ph: ProxyHandler, pr: urllib.parse.ParseResult):
    """
    Handle requests to aum path, which is used in a simple way to
    verify that one is communicating with this proxy server
    """
    import importlib
    queryParams = urllib.parse.parse_qs(pr.query)
    url = queryParams['url']
    print(f"DBUG:HandleAUM:Url:{url}")
    url = url[0]
    if (not url) or (len(url) == 0):
        ph.send_error(400, f"WARN:HandleAUM:MissingUrl/UnknownQuery?!")
        return
    urlParts = url.split('.',1)
    if gAllowedCalls.get(urlParts[0], None) == None:
        ph.send_error(403, f"WARN:HandleAUM:Forbidden:{urlParts[0]}")
        return
    for dep in gAllowedCalls[urlParts[0]]:
        try:
            importlib.import_module(dep)
        except ImportError as exc:
            ph.send_error(400, f"WARN:HandleAUM:{urlParts[0]}:Support module [{dep}] missing or has issues")
            return
    print(f"INFO:HandleAUM:Availability ok for:{urlParts[0]}")
    ph.send_response_only(200, "bharatavarshe")
    ph.send_header('Access-Control-Allow-Origin', '*')
    ph.end_headers()


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
    import json
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
    print(json.dumps(gMe, indent=4))
    for k in gConfigNeeded:
        if gMe.get(k) == None:
            print(f"ERRR:ProcessArgs:{k}:missing, did you forget to pass the config file...")
            exit(104)
    mDebug.setup(gMe['--debug'])
    uv.validator_setup(gMe['--allowed.schemes'], gMe['--allowed.domains'])


def setup_server():
    """
    Helps setup a http/https server
    """
    try:
        gMe['serverAddr'] = ('', gMe['--port'])
        gMe['server'] = http.server.HTTPServer(gMe['serverAddr'], ProxyHandler)
        if gMe.get('--sec.keyfile') and gMe.get('--sec.certfile'):
            sslCtxt = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            sslCtxt.load_cert_chain(certfile=gMe['--sec.certfile'], keyfile=gMe['--sec.keyfile'])
            sslCtxt.minimum_version = ssl.TLSVersion.MAXIMUM_SUPPORTED
            sslCtxt.maximum_version = ssl.TLSVersion.MAXIMUM_SUPPORTED
            gMe['server'].socket = sslCtxt.wrap_socket(gMe['server'].socket, server_side=True)
            print(f"INFO:SetupServer:Starting on {gMe['serverAddr']}:Https mode")
        else:
            print(f"INFO:SetupServer:Starting on {gMe['serverAddr']}:Http mode")
    except Exception as exc:
        print(f"ERRR:SetupServer:{traceback.format_exc()}")
        raise RuntimeError(f"SetupServer:{exc}") from exc


def run():
    try:
        setup_server()
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
