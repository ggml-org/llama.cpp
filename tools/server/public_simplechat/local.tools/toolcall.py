# Tool Call Base
# by Humans for All

from typing import Any, TypeAlias
from dataclasses import dataclass
import http
import http.client
import urllib.parse


#
# A sample tool call meta
#

fetchurlraw_meta = {
        "type": "function",
        "function": {
            "name": "fetch_url_raw",
            "description": "Fetch contents of the requested url (local file path / web based) through a proxy server and return the got content as is, in few seconds. Mainly useful for getting textual non binary contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":{
                        "type":"string",
                        "description":"url of the local file / web content to fetch"
                    }
                },
                "required": ["url"]
            }
        }
    }


#
# Dataclasses to help with Tool Calls
#

TCInArgs: TypeAlias = dict[str, Any]

@dataclass
class TCInProperty():
    type: str
    description: str

TCInProperties: TypeAlias = dict[str, TCInProperty]

@dataclass
class TCInParameters():
    type: str = "object"
    properties: TCInProperties = {}
    required: list[str] = []

@dataclass
class TCFunction():
    name: str
    description: str
    parameters: TCInParameters

@dataclass
class ToolCallMeta():
    type: str = "function"
    function: TCFunction|None = None

@dataclass
class TollCallResponse():
    status: bool
    tcid: str
    name: str
    content: str = ""

@dataclass(frozen=True)
class TCOutResponse:
    """
    Used to return result from tool call.
    """
    callOk: bool
    statusCode: int
    statusMsg: str = ""
    contentType: str = ""
    contentData: bytes = b""


@dataclass
class ToolCall():
    name: str

    def tcf_meta(self) -> TCFunction|None:
        return None

    def tc_handle(self, args: TCInArgs, inHeaders: http.client.HTTPMessage) -> TCOutResponse:
        return TCOutResponse(False, 500)

    def meta(self) -> ToolCallMeta:
        tcf = self.tcf_meta()
        return ToolCallMeta("function", tcf)

    def handler(self, callId: str, args: Any, inHeaders: http.client.HTTPMessage) -> TollCallResponse:
        got = self.tc_handle(args, inHeaders)
        return TollCallResponse(got.callOk, callId, self.name, got.contentData.decode('utf-8'))
