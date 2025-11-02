# Handle URL validation
# by Humans for All

import urllib.parse
import re
from dataclasses import dataclass


gMe = {
}


@dataclass(frozen=True)
class UrlVResponse:
    """
    Used to return result wrt urlreq helper below.
    """
    callOk: bool
    statusCode: int
    statusMsg: str = ""


def validator_ok():
    pass


def validate_url(url: str, tag: str):
    """
    Implement a re based filter logic on the specified url.
    """
    tag=f"VU:{tag}"
    if (not gMe.get('--allowed.domains')):
        return UrlVResponse(False, 400, f"DBUG:{tag}:MissingAllowedDomains")
    urlParts = urllib.parse.urlparse(url)
    print(f"DBUG:ValidateUrl:{urlParts}, {urlParts.hostname}")
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
