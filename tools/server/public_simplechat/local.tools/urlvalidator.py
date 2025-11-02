# Handle URL validation
# by Humans for All

import urllib.parse
import re
from dataclasses import dataclass


gMe = {
}


def validator_setup(allowedSchemes: list[str], allowedDomains: list[str]):
    global gMe
    gMe['--allowed.schemes'] = allowedSchemes
    gMe['--allowed.domains'] = allowedDomains


@dataclass(frozen=True)
class UrlVResponse:
    """
    Used to return detailed results below.
    """
    callOk: bool
    statusCode: int
    statusMsg: str = ""


def validator_ok(tag: str):
    """
    Cross check validator is setup as needed
    """
    if (not gMe.get('--allowed.domains')):
        return UrlVResponse(False, 400, f"DBUG:{tag}:MissingAllowedDomains")
    if (not gMe.get('--allowed.schemes')):
        return UrlVResponse(False, 400, f"DBUG:{tag}:MissingAllowedSchemes")
    return UrlVResponse(True, 100)


def validate_fileurl(urlParts: urllib.parse.ParseResult, tag: str):
    if urlParts.netloc != '':
        return UrlVResponse(False, 400, f"WARN:{tag}:Malformed file url")
    return UrlVResponse(True, 100)


def validate_weburl(urlParts: urllib.parse.ParseResult, tag: str):
    # Cross check hostname
    urlHName = urlParts.hostname
    if not urlHName:
        return UrlVResponse(False, 400, f"WARN:{tag}:Missing hostname in Url")
    bMatched = False
    for filter in gMe['--allowed.domains']:
        if re.match(filter, urlHName):
            bMatched = True
    if not bMatched:
        return UrlVResponse(False, 400, f"WARN:{tag}:requested hostname not allowed")
    return UrlVResponse(True, 200)


def validate_url(url: str, tag: str):
    """
    Implement a re based filter logic on the specified url.
    """
    tag=f"VU:{tag}"
    vok = validator_ok(tag)
    if (not vok.callOk):
        return vok
    if (not url):
        return UrlVResponse(False, 400, f"WARN:{tag}:Missing url")
    urlParts = urllib.parse.urlparse(url)
    print(f"DBUG:{tag}:{urlParts}, {urlParts.hostname}")
    # Cross check scheme
    urlScheme = urlParts.scheme
    if not urlScheme:
        return UrlVResponse(False, 400, f"WARN:{tag}:Missing scheme in Url")
    if not (urlScheme in gMe['--allowed.schemes']):
        return UrlVResponse(False, 400, f"WARN:{tag}:requested scheme not allowed")
    if urlScheme == 'file':
        return validate_fileurl(urlParts, tag)
    else:
        return validate_weburl(urlParts, tag)
