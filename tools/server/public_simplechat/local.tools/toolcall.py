# Tool Call Base
# by Humans for All

from typing import Any, TypeAlias
from dataclasses import dataclass


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


class ToolCall():

    name: str = ""

    def tcf_meta(self) -> TCFunction|None:
        return None

    def tc_handle(self, args: TCInProperties) -> tuple[bool, str]:
        return (False, "")

    def meta(self) -> ToolCallMeta:
        tcf = self.tcf_meta()
        return ToolCallMeta("function", tcf)

    def handler(self, callId: str, args: Any) -> TollCallResponse:
        got = self.tc_handle(args)
        return TollCallResponse(got[0], callId, self.name, got[1])
