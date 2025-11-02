# Helper to manage web related requests
# by Humans for All

import urllib.parse
import urllib.request
import simpleproxy as root
import urlvalidator as uv
from dataclasses import dataclass
import html.parser


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


def handle_urlreq(ph: root.ProxyHandler, pr: urllib.parse.ParseResult, tag: str):
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
            root.debug_dump({ 'url': req.full_url, 'headers': req.headers, 'ctype': contentType }, { 'cdata': contentData })
        return UrlReqResp(True, statusCode, "", contentType, contentData)
    except Exception as exc:
        return UrlReqResp(False, 502, f"WARN:{tag}:Failed:{exc}")


def handle_urlraw(ph: root.ProxyHandler, pr: urllib.parse.ParseResult):
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


def handle_urltext(ph: root.ProxyHandler, pr: urllib.parse.ParseResult):
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
        root.debug_dump({ 'RawText': 'yes', 'StrippedText': 'yes' }, { 'RawText': textHtml.text, 'StrippedText': textHtml.get_stripped_text() })
    except Exception as exc:
        ph.send_error(502, f"WARN:UrlTextFailed:{exc}")
