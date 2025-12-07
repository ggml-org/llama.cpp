# Helper to manage web related requests
# by Humans for All

import urlvalidator as uv
import html.parser
import debug
import filemagic as mFile
import json
import re
import urllib.parse
from typing import Any, cast, Optional
from dataclasses import dataclass
import toolcalls as mTC



def handle_urlreq(url: str, inHeaders: mTC.HttpHeaders, tag: str):
    """
    Common part of the url request handling used by both urlraw and urltext.

    Verify the url being requested is allowed.

    Fetch the requested url.
    """
    tag=f"UrlReq:{tag}"
    print(f"DBUG:{tag}:Url:{url}")
    gotVU = uv.validate_url(url, tag)
    if not gotVU.callOk:
        return mTC.TCOutResponse(gotVU.callOk, gotVU.statusCode, gotVU.statusMsg)
    try:
        # Get requested url
        return mFile.get_file(url, tag, "text/html", inHeaders)
    except Exception as exc:
        return mTC.TCOutResponse(False, 502, f"WARN:{tag}:Failed:{exc}")


class TCUrlRaw(mTC.ToolCall):

    def tcf_meta(self) -> mTC.TCFunction:
        return mTC.TCFunction(
            self.name,
            "Fetch contents of the requested url (local file path / web based) through a proxy server and return the got content as is, in few seconds. Mainly useful for getting textual non binary contents",
            mTC.TCInParameters(
                "object",
                {
                    "url": mTC.TCInProperty(
                        "string",
                        "url of the local file / web content to fetch"
                    )
                },
                [ "url" ]
            )
        )

    def tc_handle(self, args: mTC.TCInArgs, inHeaders: mTC.HttpHeaders) -> mTC.TCOutResponse:
        try:
            # Get requested url
            got = handle_urlreq(args['url'], inHeaders, "HandleTCUrlRaw")
            return got
        except Exception as exc:
            return mTC.TCOutResponse(False, 502, f"WARN:UrlRaw:Failed:{exc}")


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


gTagDropsHTMLTextSample = [ { 'tag': 'div', 'id': "header" } ]

class TCHtmlText(mTC.ToolCall):

    def tcf_meta(self) -> mTC.TCFunction:
        return mTC.TCFunction(
            self.name,
            "Fetch html content from given url through a proxy server and return its text content after stripping away html tags as well as uneeded blocks like head, script, style, header, footer, nav in few seconds",
            mTC.TCInParameters(
                "object",
                {
                    "url": mTC.TCInProperty(
                        "string",
                        "url of the html page that needs to be fetched and inturn unwanted stuff stripped from its contents to an extent"
                    ),
                    "tagDrops": mTC.TCInProperty(
                        "string",
                        (
                            "Optionally specify a json stringified list of tag-and-id dicts of tag blocks to drop from html."
                            "For each tag block that needs to be dropped, one needs to specify the tag type and its associated id attribute."
                            "where the tag types (ie div, span, p, a ...) are always mentioned in lower case."
                            f"For example when fetching a search web site, one could use {json.dumps(gTagDropsHTMLTextSample)} and so..."
                        )
                    )
                },
                [ "url" ]
            )
        )

    def tc_handle(self, args: mTC.TCInArgs, inHeaders: mTC.HttpHeaders) -> mTC.TCOutResponse:
        try:
            # Get requested url
            got = handle_urlreq(args['url'], inHeaders, "HandleTCHtmlText")
            if not got.callOk:
                return got
            # Extract Text
            tagDrops = args.get('tagDrops')
            if not tagDrops:
                tagDrops = []
            else:
                tagDrops = cast(list[dict[str,Any]], json.loads(tagDrops))
            textHtml = TextHtmlParser(tagDrops)
            textHtml.feed(got.contentData.decode('utf-8'))
            debug.dump({ 'op': 'MCPWeb.HtmlText', 'RawText': 'yes', 'StrippedText': 'yes' }, { 'RawText': textHtml.text, 'StrippedText': textHtml.get_stripped_text() })
            return mTC.TCOutResponse(True, got.statusCode, got.statusMsg, got.contentType, textHtml.get_stripped_text().encode('utf-8'))
        except Exception as exc:
            return mTC.TCOutResponse(False, 502, f"WARN:HtmlText:Failed:{exc}")



@dataclass(frozen=True)
class SearchEngine:
    template: str
    drop: Optional[list[dict[str, str]]] = None

#Few web search engine url template strings.
#The SEARCHWORDS keyword will get replaced by the actual user specified search words at runtime.
gSearchEngines: dict[str, SearchEngine] = {
    "duckduckgo": SearchEngine(
        "https://duckduckgo.com/html/?q=SEARCHWORDS",
        [ { 'tag': 'div', 'id': "header" } ]
    ),
    "_bing": SearchEngine(
        "https://www.bing.com/search?q=SEARCHWORDS"  # doesnt seem to like google chrome clients in particular
    ),
    "brave": SearchEngine(
        "https://search.brave.com/search?q=SEARCHWORDS",
    ),
    "_google": SearchEngine(
        "https://www.google.com/search?q=SEARCHWORDS", # doesnt seem to like any client in general
    ),
}

class TCSearchWeb(mTC.ToolCall):

    def tcf_meta(self) -> mTC.TCFunction:
        return mTC.TCFunction(
            self.name,
            "Search web for given words and return plain text content after stripping html tags as well as head, script, style, header, footer, nav blocks from got html result page, in few seconds",
            mTC.TCInParameters(
                "object",
                {
                    "words": mTC.TCInProperty (
                        "string",
                        "The words to search for on the web"
                    ),
                    "searchEngine": mTC.TCInProperty(
                        "string",
                        f"Name of the search engine to use. The supported search engines are {list(gSearchEngines.keys())}. The engine names prefixed with _ may not work many a times"
                    )
                },
                [ "words", "searchEngine" ]
            )
        )

    def tc_handle(self, args: mTC.TCInArgs, inHeaders: mTC.HttpHeaders) -> mTC.TCOutResponse:
        try:
            words = args['words']
            engineName = args['searchEngine']
            if not engineName:
                engineName = list(gSearchEngines.keys())[0]
            searchEngine = gSearchEngines[engineName]
            searchUrl = searchEngine.template.replace("SEARCHWORDS", urllib.parse.quote(words, safe=''))
            # Get requested url
            got = handle_urlreq(searchUrl, inHeaders, "HandleTCSearchWeb")
            if not got.callOk:
                return got
            # Extract Text
            tagDrops = searchEngine.drop
            if not tagDrops:
                tagDrops = []
            textHtml = TextHtmlParser(tagDrops)
            textHtml.feed(got.contentData.decode('utf-8'))
            debug.dump({ 'op': 'MCPWeb.SearchWeb', 'RawText': 'yes', 'StrippedText': 'yes' }, { 'RawText': textHtml.text, 'StrippedText': textHtml.get_stripped_text() })
            return mTC.TCOutResponse(True, got.statusCode, got.statusMsg, got.contentType, textHtml.get_stripped_text().encode('utf-8'))
        except Exception as exc:
            return mTC.TCOutResponse(False, 502, f"WARN:SearchWeb:Failed:{exc}")



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


gRSSTagDropREsDefault = [
    "^rss:channel:item:guid:.*",
    "^rss:channel:item:link:.*",
    "^rss:channel:item:description:.*",
    ".*:image:.*",
    ".*:enclosure:.*"
]

class TCXmlFiltered(mTC.ToolCall):

    def tcf_meta(self) -> mTC.TCFunction:
        return mTC.TCFunction(
            self.name,
            "Fetch requested xml url through a proxy server that can optionally filter out unwanted tags and their contents. Will take few seconds",
            mTC.TCInParameters(
                "object",
                {
                    "url": mTC.TCInProperty(
                        "string",
                        "url of the xml file that will be fetched"
                    ),
                    "tagDropREs": mTC.TCInProperty(
                        "string",
                        (
                            "Optionally specify a json stringified list of xml tag heirarchies to drop."
                            "For each tag that needs to be dropped, one needs to specify regular expression of the heirarchy of tags involved,"
                            "where the tag names are always mentioned in lower case along with a : as suffix."
                            f"For example for rss feeds one could use {json.dumps(gRSSTagDropREsDefault)} and so..."
                        )
                    )
                },
                [ "url" ]
            )
        )

    def tc_handle(self, args: mTC.TCInArgs, inHeaders: mTC.HttpHeaders) -> mTC.TCOutResponse:
        try:
            # Get requested url
            got = handle_urlreq(args['url'], inHeaders, "HandleTCXMLFiltered")
            if not got.callOk:
                return got
            # Extract Text
            tagDropREs = args.get('tagDropREs')
            if not tagDropREs:
                tagDropREs = []
            else:
                tagDropREs = cast(list[str], json.loads(tagDropREs))
            xmlFiltered = XMLFilterParser(tagDropREs)
            xmlFiltered.feed(got.contentData.decode('utf-8'))
            debug.dump({ 'op': 'MCPWeb.XMLFiltered' }, { 'RawText': xmlFiltered.text })
            return mTC.TCOutResponse(True, got.statusCode, got.statusMsg, got.contentType, xmlFiltered.text.encode('utf-8'))
        except Exception as exc:
            return mTC.TCOutResponse(False, 502, f"WARN:XMLFiltered:Failed:{exc}")


def ok():
    return True
