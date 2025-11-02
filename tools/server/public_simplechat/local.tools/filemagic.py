# Handle file related helpers, be it a local file or one on the internet
# by Humans for All

import urllib.request
import urllib.parse
import debug
from dataclasses import dataclass


@dataclass(frozen=True)
class Response:
    """
    Used to return result wrt urlreq helper below.
    """
    callOk: bool
    statusCode: int
    statusMsg: str = ""
    contentType: str = ""
    contentData: bytes = b""



def get_from_web(url: str, tag: str, inContentType: str, inHeaders: dict[str, str|None]):
    """
    Get the url specified from web.

    If passed header doesnt contain certain useful http header entries,
    some predefined defaults will be used in place.
    """
    try:
        hUA = inHeaders.get('User-Agent', None)
        if not hUA:
            hUA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0'
        hAL = inHeaders.get('Accept-Language', None)
        if not hAL:
            hAL = "en-US,en;q=0.9"
        hA = inHeaders.get('Accept', None)
        if not hA:
            hA = "text/html,*/*"
        headers = {
            'User-Agent': hUA,
            'Accept': hA,
            'Accept-Language': hAL
        }
        req = urllib.request.Request(url, headers=headers)
        # Get requested url
        print(f"DBUG:{tag}:Req:{req.full_url}:{req.headers}")
        with urllib.request.urlopen(req, timeout=10) as response:
            contentData = response.read()
            statusCode = response.status or 200
            contentType = response.getheader('Content-Type') or inContentType
            debug.dump({ 'url': req.full_url, 'headers': req.headers, 'ctype': contentType }, { 'cdata': contentData })
        return Response(True, statusCode, "", contentType, contentData)
    except Exception as exc:
        return Response(False, 502, f"WARN:{tag}:Failed:{exc}")


def get_from_local(urlParts: urllib.parse.ParseResult, tag: str, inContentType: str):
    """
    Get the requested file from the local filesystem
    """
    try:
        fPdf = open(urlParts.path, 'rb')
        dPdf = fPdf.read()
        return Response(True, 200, "", inContentType, dPdf)
    except Exception as exc:
        return Response(False, 502, f"WARN:{tag}:Failed:{exc}")


def get_file(url: str, tag: str, inContentType: str, inHeaders: dict[str, str|None]={}):
    """
    Based on the scheme specified in the passed url,
    either get from local file system or from the web.
    """
    urlParts = urllib.parse.urlparse(url)
    if urlParts.scheme == "file":
        return get_from_local(urlParts, tag, inContentType)
    else:
        return get_from_web(url, tag, inContentType, inHeaders)
