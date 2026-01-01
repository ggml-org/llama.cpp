unit CommandBuffer;

interface

uses
  System.SysUtils, System.Classes, System.SyncObjs, System.Generics.Collections,
  System.JSON;

const
  BUFFER_DURATION_MS = 2000; // 2 seconds

type
  TCommandRecord = record
    Timestamp: Int64;
    Group: string;
    Verb: string;
    Subject: string;
    Params: TJSONObject;
    Executed: Boolean;
    function Clone: TCommandRecord;
  end;

  TCommandBuffer = class
  private
    FLock: TCriticalSection;
    FCommandHistory: TList<TCommandRecord>;
    procedure CleanupOldCommands(ACurrentTimestamp: Int64);
  public
    constructor Create;
    destructor Destroy; override;
    procedure AddCommand(const ARecord: TCommandRecord);
    procedure RollbackToTimestamp(ATimestamp: Int64; AOnUndoCallback: TProc<TCommandRecord>);
    procedure MarkAsExecuted(ATimestamp: Int64);
    function GetCommandCount: Integer;
    procedure Clear;
  end;

implementation

{ TCommandRecord }

function TCommandRecord.Clone: TCommandRecord;
begin
  Result.Timestamp := Self.Timestamp;
  Result.Group := Self.Group;
  Result.Verb := Self.Verb;
  Result.Subject := Self.Subject;
  if Assigned(Self.Params) then
    Result.Params := TJSONObject(Self.Params.Clone)
  else
    Result.Params := nil;
  Result.Executed := Self.Executed;
end;

{ TCommandBuffer }

constructor TCommandBuffer.Create;
begin
  inherited Create;
  FLock := TCriticalSection.Create;
  FCommandHistory := TList<TCommandRecord>.Create;
end;

destructor TCommandBuffer.Destroy;
var
  I: Integer;
begin
  FLock.Enter;
  try
    // Free all JSON objects
    for I := 0 to FCommandHistory.Count - 1 do
    begin
      if Assigned(FCommandHistory[I].Params) then
        FCommandHistory[I].Params.Free;
    end;
    FCommandHistory.Free;
  finally
    FLock.Leave;
  end;
  FLock.Free;
  inherited;
end;

procedure TCommandBuffer.CleanupOldCommands(ACurrentTimestamp: Int64);
var
  I: Integer;
  CutoffTime: Int64;
begin
  // Remove commands older than 2 seconds
  CutoffTime := ACurrentTimestamp - BUFFER_DURATION_MS;
  
  for I := FCommandHistory.Count - 1 downto 0 do
  begin
    if FCommandHistory[I].Timestamp < CutoffTime then
    begin
      if Assigned(FCommandHistory[I].Params) then
        FCommandHistory[I].Params.Free;
      FCommandHistory.Delete(I);
    end;
  end;
end;

procedure TCommandBuffer.AddCommand(const ARecord: TCommandRecord);
var
  Rec: TCommandRecord;
begin
  FLock.Enter;
  try
    Rec := ARecord.Clone;
    FCommandHistory.Add(Rec);
    CleanupOldCommands(ARecord.Timestamp);
  finally
    FLock.Leave;
  end;
end;

procedure TCommandBuffer.RollbackToTimestamp(ATimestamp: Int64; AOnUndoCallback: TProc<TCommandRecord>);
var
  I: Integer;
  CommandsToUndo: TList<TCommandRecord>;
begin
  CommandsToUndo := TList<TCommandRecord>.Create;
  try
    FLock.Enter;
    try
      // Collect commands that need to be undone (executed commands after timestamp)
      for I := FCommandHistory.Count - 1 downto 0 do
      begin
        if (FCommandHistory[I].Timestamp > ATimestamp) and FCommandHistory[I].Executed then
        begin
          CommandsToUndo.Add(FCommandHistory[I].Clone);
          // Mark as not executed
          FCommandHistory.List[I].Executed := False;
        end;
      end;
    finally
      FLock.Leave;
    end;
    
    // Execute undo callbacks outside the lock to avoid deadlocks
    if Assigned(AOnUndoCallback) then
    begin
      for I := 0 to CommandsToUndo.Count - 1 do
      begin
        AOnUndoCallback(CommandsToUndo[I]);
      end;
    end;
    
    // Cleanup
    for I := 0 to CommandsToUndo.Count - 1 do
    begin
      if Assigned(CommandsToUndo[I].Params) then
        CommandsToUndo[I].Params.Free;
    end;
  finally
    CommandsToUndo.Free;
  end;
end;

procedure TCommandBuffer.MarkAsExecuted(ATimestamp: Int64);
var
  I: Integer;
begin
  FLock.Enter;
  try
    for I := 0 to FCommandHistory.Count - 1 do
    begin
      if FCommandHistory[I].Timestamp = ATimestamp then
      begin
        FCommandHistory.List[I].Executed := True;
        Break;
      end;
    end;
  finally
    FLock.Leave;
  end;
end;

function TCommandBuffer.GetCommandCount: Integer;
begin
  FLock.Enter;
  try
    Result := FCommandHistory.Count;
  finally
    FLock.Leave;
  end;
end;

procedure TCommandBuffer.Clear;
var
  I: Integer;
begin
  FLock.Enter;
  try
    for I := 0 to FCommandHistory.Count - 1 do
    begin
      if Assigned(FCommandHistory[I].Params) then
        FCommandHistory[I].Params.Free;
    end;
    FCommandHistory.Clear;
  finally
    FLock.Leave;
  end;
end;

end.
