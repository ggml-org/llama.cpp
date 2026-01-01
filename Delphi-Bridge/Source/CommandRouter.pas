unit CommandRouter;

interface

uses
  System.SysUtils, System.Classes, System.JSON, System.Net.Socket, System.Threading,
  PoseBuffer, CommandBuffer, ExportedFunctions;

type
  TCommandRouter = class
  private
    FServerSocket: TSocket;
    FListenThread: ITask;
    FActive: Boolean;
    FPort: Integer;
    FPoseBuffer: TAIPoseBuffer;
    FCommandBuffer: TCommandBuffer;
    FOnCommandReceived: TProc<string, string, string>; // Group, Verb, Intent description
    procedure ListenForConnections;
    procedure HandleClient(AClientSocket: TSocket);
    procedure ProcessCommand(const AJsonStr: string);
    procedure UndoCommand(const ACommand: TCommandRecord);
  public
    constructor Create(APort: Integer = 9001);
    destructor Destroy; override;
    procedure Start;
    procedure Stop;
    function RouteCommand(const AJson: string): Boolean;
    property PoseBuffer: TAIPoseBuffer read FPoseBuffer;
    property CommandBuffer: TCommandBuffer read FCommandBuffer;
    property OnCommandReceived: TProc<string, string, string> read FOnCommandReceived write FOnCommandReceived;
  end;

implementation

uses
  System.SyncObjs;

{ TCommandRouter }

constructor TCommandRouter.Create(APort: Integer);
begin
  inherited Create;
  FPort := APort;
  FActive := False;
  FPoseBuffer := TAIPoseBuffer.Create;
  FCommandBuffer := TCommandBuffer.Create;
end;

destructor TCommandRouter.Destroy;
begin
  Stop;
  FPoseBuffer.Free;
  FCommandBuffer.Free;
  inherited;
end;

procedure TCommandRouter.Start;
begin
  if FActive then
    Exit;
    
  FActive := True;
  
  // Start listening thread
  FListenThread := TTask.Run(
    procedure
    begin
      ListenForConnections;
    end);
end;

procedure TCommandRouter.Stop;
begin
  if not FActive then
    Exit;
    
  FActive := False;
  
  if Assigned(FServerSocket) then
  begin
    FServerSocket.Close;
    FServerSocket := nil;
  end;
  
  // Wait for thread to finish
  if Assigned(FListenThread) then
  begin
    TTask.WaitForAll([FListenThread]);
    FListenThread := nil;
  end;
end;

procedure TCommandRouter.ListenForConnections;
var
  ClientSocket: TSocket;
begin
  try
    FServerSocket := TSocket.Create(TSocketType.TCP);
    FServerSocket.Listen('', '', FPort, 5);
    
    while FActive do
    begin
      try
        ClientSocket := FServerSocket.Accept(1000); // 1 second timeout
        if Assigned(ClientSocket) then
        begin
          TTask.Run(
            procedure
            begin
              HandleClient(ClientSocket);
            end);
        end;
      except
        // Timeout or error, continue if still active
      end;
    end;
  except
    on E: Exception do
    begin
      // Log error
    end;
  end;
end;

procedure TCommandRouter.HandleClient(AClientSocket: TSocket);
var
  Buffer: TBytes;
  ReceivedData: string;
  BytesRead: Integer;
begin
  try
    SetLength(Buffer, 4096);
    ReceivedData := '';
    
    while FActive do
    begin
      BytesRead := AClientSocket.Receive(Buffer);
      if BytesRead > 0 then
      begin
        ReceivedData := ReceivedData + TEncoding.UTF8.GetString(Buffer, 0, BytesRead);
        
        // Process complete JSON objects (assuming one per line)
        while Pos(#10, ReceivedData) > 0 do
        begin
          var LineEnd := Pos(#10, ReceivedData);
          var JsonLine := Copy(ReceivedData, 1, LineEnd - 1).Trim;
          Delete(ReceivedData, 1, LineEnd);
          
          if JsonLine <> '' then
            ProcessCommand(JsonLine);
        end;
      end
      else
        Break; // Connection closed
    end;
  finally
    AClientSocket.Close;
  end;
end;

procedure TCommandRouter.ProcessCommand(const AJsonStr: string);
begin
  RouteCommand(AJsonStr);
end;

function TCommandRouter.RouteCommand(const AJson: string): Boolean;
var
  JsonObj: TJSONObject;
  Timestamp: Int64;
  Group, Verb, Subject: string;
  Params: TJSONObject;
  CommandRec: TCommandRecord;
  VerbPChar, ParamsPChar: PChar;
  ParamsStr: string;
  IntentDescription: string;
begin
  Result := False;
  JsonObj := nil;
  
  try
    // Parse JSON
    JsonObj := TJSONObject.ParseJSONValue(AJson) as TJSONObject;
    if not Assigned(JsonObj) then
      Exit;
      
    // Extract fields
    Timestamp := JsonObj.GetValue<Int64>('timestamp', 0);
    Group := JsonObj.GetValue<string>('group', '');
    Verb := JsonObj.GetValue<string>('verb', '');
    Subject := JsonObj.GetValue<string>('subject', '');
    
    if JsonObj.TryGetValue<TJSONObject>('params', Params) then
      Params := TJSONObject(Params.Clone)
    else
      Params := TJSONObject.Create;
      
    try
      // Check for race conditions - rollback if needed
      FCommandBuffer.RollbackToTimestamp(Timestamp, UndoCommand);
      FPoseBuffer.RollbackToTimestamp(Timestamp);
      
      // Create command record
      CommandRec.Timestamp := Timestamp;
      CommandRec.Group := Group;
      CommandRec.Verb := Verb;
      CommandRec.Subject := Subject;
      CommandRec.Params := TJSONObject(Params.Clone);
      CommandRec.Executed := False;
      
      // Add to command buffer before execution
      FCommandBuffer.AddCommand(CommandRec);
      
      // Build intent description for UI
      IntentDescription := Format('%s: %s %s', [Group, Verb, Subject]);
      
      // Route based on group
      if Group = 'CAMERA_CONTROL' then
      begin
        VerbPChar := PChar(Verb);
        ParamsStr := Params.ToJSON;
        ParamsPChar := PChar(ParamsStr);
        FF_HandleCameraCommand(VerbPChar, ParamsPChar);
        Result := True;
      end
      else if Group = 'ACTOR_POSE' then
      begin
        // Extract pose_data array and update pose buffer
        var PoseArray: TJSONArray;
        if Params.TryGetValue<TJSONArray>('pose_data', PoseArray) then
        begin
          FPoseBuffer.UpdatePose(Timestamp, PoseArray);
          Result := True;
        end;
      end
      else if Group = 'OBJECT_MGMT' then
      begin
        VerbPChar := PChar(Verb);
        ParamsStr := Params.ToJSON;
        ParamsPChar := PChar(ParamsStr);
        FF_HandleObjectCommand(VerbPChar, ParamsPChar);
        Result := True;
      end;
      
      // Mark as executed
      if Result then
      begin
        FCommandBuffer.MarkAsExecuted(Timestamp);
        
        // Notify UI if callback is set
        if Assigned(FOnCommandReceived) then
          FOnCommandReceived(Group, Verb, IntentDescription);
      end;
      
    finally
      if Assigned(CommandRec.Params) then
        CommandRec.Params.Free;
      Params.Free;
    end;
    
  except
    on E: Exception do
    begin
      // Log error
      Result := False;
    end;
  end;
  
  if Assigned(JsonObj) then
    JsonObj.Free;
end;

procedure TCommandRouter.UndoCommand(const ACommand: TCommandRecord);
var
  VerbPChar, ParamsPChar: PChar;
  ParamsStr: string;
  UndoVerb: string;
begin
  // Implement undo logic by calling the appropriate handlers with inverted commands
  try
    if ACommand.Group = 'CAMERA_CONTROL' then
    begin
      // Invert the command (e.g., PAN left -> PAN right)
      UndoVerb := 'UNDO_' + ACommand.Verb;
      VerbPChar := PChar(UndoVerb);
      ParamsStr := ACommand.Params.ToJSON;
      ParamsPChar := PChar(ParamsStr);
      FF_HandleCameraCommand(VerbPChar, ParamsPChar);
    end
    else if ACommand.Group = 'OBJECT_MGMT' then
    begin
      // Invert the command
      UndoVerb := 'UNDO_' + ACommand.Verb;
      VerbPChar := PChar(UndoVerb);
      ParamsStr := ACommand.Params.ToJSON;
      ParamsPChar := PChar(ParamsStr);
      FF_HandleObjectCommand(VerbPChar, ParamsPChar);
    end;
    // Note: ACTOR_POSE rollback is handled directly by the PoseBuffer
  except
    on E: Exception do
    begin
      // Log undo error
    end;
  end;
end;

end.
