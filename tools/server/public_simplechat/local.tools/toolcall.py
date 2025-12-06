# Tool Call Base
# by Humans for All

from typing import Any, TypeAlias, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import email.message


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
class ToolCallResponseEx():
    tcid: str
    name: str
    response: TCOutResponse

@dataclass
class ToolCallResponse():
    status: bool
    tcid: str
    name: str
    content: str = ""

HttpHeaders: TypeAlias = dict[str, str] | email.message.Message[str, str]


@dataclass
class ToolCall():
    name: str

    def tcf_meta(self) -> TCFunction|None:
        return None

    def tc_handle(self, args: TCInArgs, inHeaders: HttpHeaders) -> TCOutResponse:
        return TCOutResponse(False, 500)

    def meta(self) -> ToolCallMeta:
        tcf = self.tcf_meta()
        return ToolCallMeta("function", tcf)


class ToolManager():

    def __init__(self) -> None:
        self.toolcalls: dict[str, ToolCall] = {}

    def tc_add(self, fName: str, tc: ToolCall):
        self.toolcalls[fName] = tc

    def meta(self):
        oMeta = {}
        for tcName in self.toolcalls.keys():
            oMeta[tcName] = self.toolcalls[tcName].meta()

    def tc_handle(self, tcName: str, callId: str, tcArgs: TCInArgs, inHeaders: HttpHeaders) -> ToolCallResponseEx:
        try:
            response = self.toolcalls[tcName].tc_handle(tcArgs, inHeaders)
            return ToolCallResponseEx(callId, tcName, response)
        except KeyError:
            return ToolCallResponseEx(callId, tcName, TCOutResponse(False, 400, "Unknown tool call"))
