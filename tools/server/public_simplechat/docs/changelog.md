# Progress

by Humans for All.

Look into source files and git logs for the details, this is a partial changelog of stuff already done
and some of the things that one may look at in the future.

## Done

Tool Calling support added, along with a bunch of useful tool calls as well as a bundled simple proxy
if one wants to access web as part of tool call usage.

Reasoning / thinking response from Ai Models is shown to the user, as they are being generated/shared.

Chat Messages/Session and UI handling have been moved into corresponding Classes to an extent, this
helps ensure that
* switching chat sessions or loading a previous auto saved chat session will restore state including
  ui such that end user can continue the chat session from where they left it, even if in the middle
  of a tool call handshake.
* new fields added to http handshake in oneshot or streaming mode can be handled in a structured way
  to an extent.

Chat message parts seperated out and tagged to allow theming chat message as needed in future.
The default Chat UI theme/look changed to help differentiate between different messages in chat
history as well as the parts of each message in a slightly better manner. Change the theme slightly
between normal and print views (beyond previous infinite height) for better printed chat history.

A builtin data store related tool calls, inturn built on browser's indexedDB, without needing any
proxy / additional helper to handle the store. One could use the ai assistant to store ones (ie end
users) own data or data of ai model.

Trap http response errors and inform user the specific error returned by ai server.

Initial go at a pdftext tool call. It allows web / local pdf files to be read and their text content
extracted and passed to ai model for further processing, as decided by ai and end user. One could
either work with the full pdf or a subset of adjacent pages.

SimpleProxy updates
* Convert from a single monolithic file into a collection of modules.
* UrlValidator to cross check scheme and domain of requested urls,
  the whitelist inturn picked from config json
* Helpers to fetch file from local file system or the web, transparently
* Help check for needed modules before a particular service path is acknowledged as available
  through /aum service path
* urltext and related - logic to drop contents of specified tag with a given id
  * allow its use for the web search tool flow
    * setup wrt default duckduckgo search result urltext plain text cleanup and found working.
  * this works properly only if the html being processed has proper opening and ending tags
    around the area of interest.
  * remember to specify non overlapping tag blocks, if more than one specified for dropping.
    * this path not tested, but should logically work

Settings/Config default changes

* Chances are for ai models which dont support tool calling, things will be such that the tool calls
meta data shared will be silently ignored without much issue. So enabling tool calling feature by
default, so that in case one is using a ai model with tool calling the feature is readily available
for use.

* Revert SlidingWindow ChatHistory in Context from last 10 to last 5 (rather 2 more then origianl,
given more context support in todays models) by default, given that now tool handshakes go through
the tools related side channel in the http handshake and arent morphed into normal user-assistant
channel of the handshake.

* Enable CachePrompt api option given that tool calling based interactions could involve chat sessions
having ai responses built over multiple steps of tool callings etal. So independent of our client side
sliding window based drop off or even before they kick in, this can help in many cases.

UI - add ClearChat button and logic. Also add unicode icons for same as well as for Settings.

Renamed pdf_to_text to fetch_pdf_as_text so that ai model can understand the semantic better.

sys_date_time tool call has been added.

Refactor code and flow a bit wrt the client web ui
* Move the main chat related classes into its own js module file, independent of the main
  runtime entry point (rather move out the runtime entry point into its own file). This allows
  these classes to be referenced from other modules like tools related modules with full access
  to these classes's details for developers and static check tools.
* building on same make the Tools management code into a ToolsManager class which is inturn
  instantiated and the handle stored in top level Me class. This class also maintains and
  manages the web workers as well as routing of the tool calling among others.
* add a common helper for posting results directly to the main thread side web worker callback
  handlers. Inturn run the calling through a setTimeout0, so that delayed/missing response
  situation rescuer timeout logic etal flow doesnt get messed for now.

Track tool calling and inturn maintain pending tool calls so that only still valid tool call responses
will be accepted when the asynchronous tool call response is recieved. Also take care of clearing
pending tool call tracking in unhappy paths like when exception noticied as part of tool call execution,
or if there is no response within the configured timeout period.
NOTE: Currently the logic supports only 1 pending tool call per chat session.

Add support for fetch_xml_as_text tool call, fix importmaps in index.html

Renamed and updated logic wrt xml fetching to be fetch_xml_filtered. allow one to use re to identify
the tags to be filtered in a fine grained manner including filtering based on tag heirarchy
* avoid showing empty skipped tag blocks

Logic which shows the generated tool call has been updated to trap errors when parsing the function call
arguments generated by the ai. This ensures that the chat ui itself doesnt get stuck in it. Instead now
the tool call response can inform the ai model that its function call had issues.

Renamed fetch_web_url_text to fetch_html_text, so that gen ai model wont try to use this to fetch xml or
rss files, because it will return empty content, because there wont be any html content to strip the tags
and unwanted blocks before returning.

Capture the body of ai server not ok responses, to help debug as well as to show same to user.

Extract and include the outline of titles (along with calculated numbering) in the text output of pdftext
* ensure that one doesnt recurse beyond a predefined limit.

Convert NSChatMessage from typedef to Class and update ChatMessageEx, SimpleChat, MultiChatUI classes to
make use of the same.
* helpers consolidated
  * helpers to check if given instance contains reasoning or content or toolcall or tool response related
    fields/info in them.
  * helpers to get the corresponding field values
  * some of these helpers where in ChatMessageEx and beyond before
* now the network handshaked fields are declared as undefined by default (instead of empty values).
  this ensures that json stringify will automatically discard fields whose values are still undefined.
* add fields wrt tool response and update full flow to directly work with these fields instead of the
  xml based serialisation which was previously used for maintaining the different tool response fields
  within the content field (and inturn extract from there when sending to server).
  * now a dataset based attribute is used to identify when input element contains user input and when
    it contains tool call result/response.
* this simplifies the flow wrt showing chat message (also make it appear more cleanly) as well as
  identifying not yet accepted tool result and showing in user query input field and related things.
* ALERT: ON-DISK-STORAGE structure of chat sessions have changed wrt tool responses. So old saves will
  no longer work wrt tool responses

UI updates
* update logic to allow empty tool results to be sent to ai engine server
* css - when user input textarea is in tool result mode (ie wrt TOOL.TEMP role), change the background
  color to match the tool role chat message block color, so that user can easily know that the input
  area is being used for submitting tool response or user response, at any given moment in time.

Vision
* Add image_url field. Allow user to load image, which is inturn stored as a dataURL in image_url.
* when user presses submit with a message, if there is some content (image for now) in dataURL,
  then initialise image_url field with same.
* when generating chat messages for ai server network handshake, create the mixed content type of
  content field which includes both the text (from content field) and image (from image_url field)
  ie if a image_url is found wrt a image.
  * follow the openai format/template wrt these mixed content messages.
* Usage: specify a mmproj file directly or through -hf, additionally had to set --batch-size to 8k
  and ubatch-size to 2k wrt gemma3-4b-it
* when showing chat instantiate img elements to show image_urls.
  * limit horizontally to max width and vertically to 20% of the height
* show any image loaded by the user, in the corresponding image button
* consolidate dataurl handling into a bunch of helper functions.
* trap quota errors wrt localStorage etal
* dont forget to reset the file type input's value, so that reselecting the same image still
  triggers the input's change event.

SimpleChat class now allows extra fields to be specified while adding, in a generic way using a
object/literal object or equivalent.

UI Cleanup - msgs spaced out, toolcall edit hr not always, scroll ui only when required,
hide settings/info till user requests, heading gradient

iDB module
* add open, transact, put and get. Use for chat session save and load
* getKeys used to show Restore/Load button wrt chat sessions.

ChatMessage
* assign a globally unique (ie across sessions) id to each chat message instance.
* add support for deleting chat message based on its uniquie id in SimpleChat.
  * try ensure that adjacent messages remain on screen, after a message is deleted from session.
* add a popover div block in html, which acts as a popup menu containing buttons to work with
  individual chat messages.
  * experiment and finalise on anchor based relative positioning of the popover menu.
  * have a del button, which allows one to delete the currently in focus chat message.
  * have a copy button, which allows one to copy the textual content into system clipboard.

MultiChatUI
* chat_show takes care of showing or clearing tool call edit / trigger as well as tool response
  edit / submit. Also show the currently active tool call and its response before it is submitted
  was previously only shown in the edit / trigger and edit / submit ui elements, now instead it
  also shows as part of the chat session message blocks, so that user can delete or copy these
  if needed using the same mechanism as other messages in the chat session.
* use a delete msg helper, which takes care of deleting the msg from chat session as well as
  efficiently update ui to any extent by removing the corresponding element directly from existing
  chat session ui without recreating the full chat session ui.
* a helper to add a message into specified chat session, as well as show/update in the chat session
  ui by appending the chat message, instead of recreating the full chat session ui.
...

MultiChatUI+
* both chat_show and chat_uirefresh (if lastN >= 2) both take care of updating tool call edit/trigger
  as well as the tool call response edit/submit related ui elements suitably.
  * chat_show recreates currently active sliding window of chat session (which could even be full)
  * while chat_uirefresh recreates/updates ui only for the lastN messages (prefer in general, as optimal)
* normal user response / query submit as well as tool call response or error submit have been updated
  to use the optimal uirefresh logic now.

Cleanup in general
* Inform end user when loading from a saved session.
* Update starting entry point flow to avoid calling chat_show twice indirectly, inturn leading to
  two restore previously saved session blocks. Rather when adding tool calls support, and inturn
  had to account for delayed collating of available simpleproxy based tool calls, I forgot to clean
  this flow up.
* Make the sys_date_time template description bit more verbose, just in case.
* ui_userinput_reset now also resets associated Role always, inturn
  * full on version from chat_show, inturn when session switching.
    So user switchs session will reset all user input area and related data, while
    also ensuring user input area has the right needed associated role setup.
  * partial version from uirefresh, inturn adding user or tool call response messages.
* ui cleanup
  * more rounded buttons, chat messages and input area elements.
  * make the body very very lightly gray in color, while the user input area is made whiter.
  * gradients wrt heading, role-specific individual chat message blocks.
  * avoid borders and instead give a box effect through light shadows.
    * also avoid allround border around chat message role block and instead have to only one side.
  * timeout close popover menu.
  * usage notes
    * update wrt vision and toggling of sessions and system prompt through main title area.
    * fix issue with sliding window size not reflecting properly in context window entry.
  * make restore block into details based block, and anchor its position independent of db check.
  * avoid unneeded outer overall scrollbar by adjusting fullbody height in screen mode.
  * user css variable to define the overall background color and inturn use same to merge gradients
    to the background, as well as to help switch the same seemlessly between screen and print modes.
  * make the scrollbars more subtle and in the background.
  * allow user input textarea to grow vertically to some extent.
  * make things rounded across board by default. add some padding to toolcall details block, ...
  * use icons without text wrt chat sessions++, new chat, clear chat and settings top level buttons.
    * use title property/attribute to give a hint to the user about the button functionality.
  * add role specific background gradients wrt the tool call trigger and user input block as well as
    fix wrt the tool temp message block. also wrt system input block at top.
    * also rename the TEMP role tags to use -TEMP instead of .TEMP, so that CSS rule selectors will
      treat such tags like role-TOOL-TEMP as say a proper css class name rather than messing up with
      something like role-TOOL.TEMP which will get split to role-TOOL and TEMP and inturn corresponding
      css rule doesnt/wont get applied.
    * given that now there is a proper visual cue based seperation of the tool call trigger block from
      surrounding content, using proper seperate tool call specific coloring, so remove the <HR> horiz
      line seperation wrt tool call trigger block.
    * however retain the horizontal line seperation between the tool trigger block and user input block,
      given that some users and some ai dont seem to identify the difference very easily.
  * work around firefox currently not yet supporting anchor based relative positioning of popover.
  * ensure the new uirefresh flow handles the following situations in a clean way like
    * a new chat session clearing out usagenote+restore+currentconfig, as user starts chatting
    * the settings ui getting cleared out as user starts/continues chatting directly into user input
      without using chat session button to switch back to the chat.
* Auto ObjPropsEdit UI
  * allow it to be themed by assigning id to top level block.
  * fix a oversight (forgotten $) with use of templated literals and having variables in them.
  * ensure full props hierarchy is accounted for when setting the id of elements.
* Chat button to toggle sessions buttons and system prompt.
* Use unix date format markers wrt sys_date_time toolcall, also add w (day of week).
* Free up the useful vertical space by merging chat sessions buttons/tabs into heading
* Allow user to load multiple images and submit to ai as part of a single user message.
* Use popover ui to allow user to view larger versions of loaded images as well as remove before submitting
  to ai, if and when needed.
* Add external_ai toolcall with no access to internet or tool calls (helps avoid recursive ai tool calling).
  User can see response generated by the external ai tool call, as and when it is recieved.
* Maintain chat session specific DivStream elements, and show live ai responses (through corresponding
  DivStream) wrt the current chat session as well as any from the external ai tool call session.
  In future, if the logic is updated to allow switching chat session ui in the middle of a pending tool call
  or pending ai server response, things wont mess up ui, as they will be updating their respective DivStream.
  Also switching sessions takes care of showing the right DivStream ie of currently switched to chat, so that
  end user can see the streamed response from that chat session as it is occuring.
* Cleanup the tool call descriptions and verbose messages returned a bit.

Chat Session specific settings
* Needed so that one could
  * setup a different ai model / engine as the external ai backend.
  * interact with different independent ai models / engines / parallel instances in general
* Move needed configs from Me into a seperate Config class.
  * also move ShowSettings, ShowInfo etal into Config class
* SimpleChat maintains an instance of Config class instead of Me.
* ToolsManager and the different tool call modules have been updated to
  * have seperate init and setup calls.
    * init is called at the begining
    * setup will be called when ever a chat session is being created
      and or in future when ever any config of interest changes.
  * pick needed config etal from the specified chatId's config and not any global config.
* Starting flow updated to chain the different logical blocks of code
  * first allow tools manager to be initd
  * next create the needed default set of sessions, while parallely calling tool manager setup as needed.
    * ensures that the available list of tool calls match the config of the chat session involved.
      Needed as user could change tools related proxy server url.
  * next setup the main ui as needed.
* Hide user-input area and tool call validate/trigger area when switching into settings and ensure they
  get unhidden when returning back, as needed.
* Save and restore ChatSession config entries, as needed, in localStorage.
  * load previously saved config if any, when creating ChatSession
  * when ever switching, including into a, ChatSession, Configs of all chat sessions are saved.
* ALERT: If a chat session's tools proxyUrl is changed
  * the same will be picked up immidiately wrt all subsequent tool calls which depend on the
    tool call proxy server.
  * however any resultant changes to the available tool calls list wont get reflected,
    till one reloads the program.
* uirefresh helper ensures client side sliding window is always satisfied.
  * now it remove messages no longer in the sliding window, so user only sees what is sent to the ai server,
    in the chat session messages ui.
  * avoids adding additional control specifically for ui, and instead stick to the ai nw handshake related
    chat sliding window size (which takes care of try avoid overloading the ai model context size) selected
    by user already. User can always change the sliding window size to view past messages beyond the currently
    active sliding window size and then switch back again, if they want to.
* More flexibility to user wrt ExternalAi tool call ie ai calling ai
  * the user can change the default behaviour of tools being disabled and sliding window of 1
  * program restart will reset these back to the default
* Ui module cleanup to avoid duplicated/unneeded boiler plates, including using updated jsdoc annotations
* A simple minded basic Markdown to Html logic with support for below to some extent
  * headings, horiz line,
  * lists (ordered, unordered, intermixed at diff leves)
    accomodate lines without list markers inbetween list items to some extent, hopefully in a sane way.
  * tables, fenced code blocks, blockquotes
  * User given control to enable markdown implicitly at a session level, or explicitly set wrt individual msgs.
* Rename fetch_web_url_raw to fetch_url_raw, avoids confusion and matchs semantic of access to local and web.
* Now external_ai specific special chat session's and inturn external ai tool call's ai live response stream
  is visible in the chat session which triggered external ai, only till one gets respose or the tool call times
  out. In turn if the tool call times out, one can send the timeout message as the response to the tool call
  or what ever they see fit. Parallely, they can always look into the external ai specific special chat session
  tab to see the ai response live stream and the progress wrt the tool call that timed out.
* SimpleProxy
  * add ssl ie https support and restrict it to latest supported ssl/tls version
  * enable multi threaded ssl and client request handling, so that rogue clients cant mount simple DoS
    by opening connection and then missing in action.
  * switch to a Dicty DataClass based Config with better type validation and usage, instead of literal dict++
* ToolCall, ToolManager and related classes based flow wrt the tool calls.
  * all existing tool calls duplicated and updated to support and build on this new flow.
* Initial skeleton towards SimpleMCP, a mcpish server, which uses post and json rpcish based handshake flow,
  so that tool calls supported through SimpleProxy can be exposed through a MCP standardish mechanism.
  * can allow others beyond AnveshikaSallap client to use the corresponding tool calls
  * can allow AnveshikaSallap client to support other MCP servers and their exposed tool calls in future.
  Mcp command tools/list implemented and verified at a basic level
  Mcp command tools/call implemented, need to verify and update the initial go version
* Initial skeleton towards ToolMCP, a mcpish client logic
  Mcp command tools/list handshake implemented, need to verify and update this initial go
  Mcp command tools/call handshake implemented, need to verify and update this initial go
  Minimal cross check wrt tools/list and tools/call.
* MCPish and not full fledged MCP currently
  * no initialise command handshake
  * use seconds since unix epoch or toolcall id, as the case maybe, as the id wrt json-rpc calls
  * the tools/list response mirrors the openai rest api convention rather than mcp convention
    * uses the additional type: function wrapper wrt tool call meta
    * uses the keyword parameters instead of inputschema or so
* Retire the previous simpleproxy.py and its related helpers, including ones running in browser env.


## ToDo

Is the tool call promise land trap deep enough, need to think through and explore around this once later.

Add fetch_rss and may be different document formats processing related tool calling, in turn through
the simpleproxy.py if and where needed.

* Using xmlfiltered and tagDropREs of
  * ["^rss:channel:item:(?!title).+$"] one can fetch and extract out all the titles.
  * ["^rss:channel:item:(?!title|link|description).+$"] one can fetch and extract out all the
    titles along with corresponding links and descriptions
  * rather with some minimal proding and guidance gpt-oss generated this to use xmlfiltered to read rss

Add a option/button to reset the chat session config, to defaults.

Have a seperate helper to show the user input area, based on set state. And have support for multiple images
if the models support same. It should also take care of some aspects of the tool call response edit / submit,
potentially.

MAYBE add a special ClientSideOnly role for use wrt Chat history to maintain things to be shown in a chat
session to the end user, but inturn not to be sent to the ai server. Ex current settings, any edits to toolcall,
any tool call or server handshake errors seen (which user might have worked around as needed and continued the
conversation) or so ...

Updating system prompt, will reset user input area fully now, which seems a good enough behaviour, while
keeping the code flow also simple and straight, do I need to change it, I dont think so as of now.

For now amn't bringing in mozilla/github/standard-entities pdf, md, mathslatex etal javascript libraries for
their respective functionalities.

Add support for base64 encoded pdf passing to ai models, when the models and llama engine gain that capability
in turn using openai file - file-data type sub block within content array or so ...
