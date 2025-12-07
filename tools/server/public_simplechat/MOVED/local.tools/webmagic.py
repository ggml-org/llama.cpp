# Helper to manage web related requests
# by Humans for All

import urllib.parse
import urlvalidator as uv
from dataclasses import dataclass
import html.parser
import debug
import filemagic as mFile
import json
import re
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from simpleproxy import ProxyHandler



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


def handle_urlreq(ph: 'ProxyHandler', pr: urllib.parse.ParseResult, tag: str):
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
        hUA = ph.headers.get('User-Agent', None)
        hAL = ph.headers.get('Accept-Language', None)
        hA = ph.headers.get('Accept', None)
        headers = {
            'User-Agent': hUA,
            'Accept': hA,
            'Accept-Language': hAL
        }
        # Get requested url
        gotFile = mFile.get_file(url, tag, "text/html", headers)
        return UrlReqResp(gotFile.callOk, gotFile.statusCode, gotFile.statusMsg, gotFile.contentType, gotFile.contentData.decode('utf-8'))
    except Exception as exc:
        return UrlReqResp(False, 502, f"WARN:{tag}:Failed:{exc}")


def handle_urlraw(ph: 'ProxyHandler', pr: urllib.parse.ParseResult):
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

    Supports one to specify a list of tags and their corresponding id attributes, so that contents
    within such specified blocks will be dropped.

    * this works properly only if the html being processed has proper opening and ending tags
    around the area of interest.
    * remember to specify non overlapping tag blocks, if more than one specified for dropping.
        * this path not tested, but should logically work

    This helps return a relatively clean textual representation of the html file/content being parsed.
    """

    def __init__(self, tagDrops: list[dict[str, Any]]):
        super().__init__()
        self.tagDrops = tagDrops
        print(f"DBUG:TextHtmlParser:{self.tagDrops}")
        self.inside = {
            'body': False,
            'script': False,
            'style': False,
            'header': False,
            'footer': False,
            'nav': False,
        }
        self.monitored = [ 'body', 'script', 'style', 'header', 'footer', 'nav' ]
        self.bCapture = False
        self.text = ""
        self.textStripped = ""
        self.droptagType = None
        self.droptagCount = 0

    def do_capture(self):
        """
        Helps decide whether to capture contents or discard them.
        """
        if self.inside['body'] and not (self.inside['script'] or self.inside['style'] or self.inside['header'] or self.inside['footer'] or self.inside['nav'] or (self.droptagCount > 0)):
            return True
        return False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag in self.monitored:
            self.inside[tag] = True
        for tagMeta in self.tagDrops:
            if tag != tagMeta['tag']:
                continue
            if (self.droptagCount > 0) and (self.droptagType == tag):
                self.droptagCount += 1
                continue
            for attr in attrs:
                if attr[0] != 'id':
                    continue
                if attr[1] == tagMeta['id']:
                    self.droptagCount += 1
                    self.droptagType = tag
                    print(f"DBUG:THP:Start:Tag found [{tag}:{attr[1]}]...")

    def handle_endtag(self, tag: str):
        if tag in self.monitored:
            self.inside[tag] = False
        if self.droptagType and (tag == self.droptagType):
            self.droptagCount -= 1
            if self.droptagCount == 0:
                self.droptagType = None
                print("DBUG:THP:End:Tag found...")
            if self.droptagCount < 0:
                self.droptagCount = 0

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


def handle_htmltext(ph: 'ProxyHandler', pr: urllib.parse.ParseResult):
    try:
        # Get requested url
        got = handle_urlreq(ph, pr, "HandleHtmlText")
        if not got.callOk:
            ph.send_error(got.httpStatus, got.httpStatusMsg)
            return
        # Extract Text
        tagDrops = ph.headers.get('htmltext-tag-drops')
        if not tagDrops:
            tagDrops = []
        else:
            tagDrops = cast(list[dict[str,Any]], json.loads(tagDrops))
        textHtml = TextHtmlParser(tagDrops)
        textHtml.feed(got.contentData)
        # Send back to client
        ph.send_response(got.httpStatus)
        ph.send_header('Content-Type', got.contentType)
        # Add CORS for browser fetch, just in case
        ph.send_header('Access-Control-Allow-Origin', '*')
        ph.end_headers()
        ph.wfile.write(textHtml.get_stripped_text().encode('utf-8'))
        debug.dump({ 'op': 'WebMagic.HtmlText', 'RawText': 'yes', 'StrippedText': 'yes' }, { 'RawText': textHtml.text, 'StrippedText': textHtml.get_stripped_text() })
    except Exception as exc:
        ph.send_error(502, f"WARN:HtmlText:Failed:{exc}")


class XMLFilterParser(html.parser.HTMLParser):
    """
    A simple minded logic used to strip xml content of
    * unwanted tags and their contents, using re
    * this works properly only if the xml being processed has
      proper opening and ending tags around the area of interest.

    This can help return a cleaned up xml file.
    """

    def __init__(self, tagDropREs: list[str]):
        """
        tagDropREs - allows one to specify a list of tags related REs,
        to help drop the corresponding tags and their contents fully.

        To drop a tag, specify regular expression
        * that matches the corresponding heirarchy of tags involved
            * where the tag names should be in lower case and suffixed with :
        * if interested in dropping a tag independent of where it appears use
          ".*:tagname:.*" re template
        """
        super().__init__()
        self.tagDropREs = list(map(str.lower, tagDropREs))
        print(f"DBUG:XMLFilterParser:{self.tagDropREs}")
        self.text = ""
        self.prefixTags = []
        self.prefix = ""
        self.lastTrackedCB = ""

    def do_capture(self):
        """
        Helps decide whether to capture contents or discard them.
        """
        curTagH = f'{":".join(self.prefixTags)}:'
        for dropRE in self.tagDropREs:
            if re.match(dropRE, curTagH):
                return False
        return True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        self.prefixTags.append(tag)
        if not self.do_capture():
            return
        self.lastTrackedCB = "starttag"
        self.prefix += "\t"
        self.text += f"\n{self.prefix}<{tag}>"

    def handle_endtag(self, tag: str):
        if self.do_capture():
            if (self.lastTrackedCB == "endtag"):
                self.text += f"\n{self.prefix}</{tag}>"
            else:
                self.text += f"</{tag}>"
            self.lastTrackedCB = "endtag"
            self.prefix = self.prefix[:-1]
        self.prefixTags.pop()

    def handle_data(self, data: str):
        if self.do_capture():
            self.text += f"{data}"


def handle_xmlfiltered(ph: 'ProxyHandler', pr: urllib.parse.ParseResult):
    try:
        # Get requested url
        got = handle_urlreq(ph, pr, "HandleXMLFiltered")
        if not got.callOk:
            ph.send_error(got.httpStatus, got.httpStatusMsg)
            return
        # Extract Text
        tagDropREs = ph.headers.get('xmlfiltered-tagdrop-res')
        if not tagDropREs:
            tagDropREs = []
        else:
            tagDropREs = cast(list[str], json.loads(tagDropREs))
        xmlFiltered = XMLFilterParser(tagDropREs)
        xmlFiltered.feed(got.contentData)
        # Send back to client
        ph.send_response(got.httpStatus)
        ph.send_header('Content-Type', got.contentType)
        # Add CORS for browser fetch, just in case
        ph.send_header('Access-Control-Allow-Origin', '*')
        ph.end_headers()
        ph.wfile.write(xmlFiltered.text.encode('utf-8'))
        debug.dump({ 'XMLFiltered': 'yes' }, { 'RawText': xmlFiltered.text })
    except Exception as exc:
        ph.send_error(502, f"WARN:XMLFiltered:Failed:{exc}")
