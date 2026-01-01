unit PoseBuffer;

interface

uses
  System.SysUtils, System.Classes, System.SyncObjs, System.Generics.Collections,
  System.JSON;

type
  TPoseData = record
    Timestamp: Int64;
    PoseArray: TJSONArray;
    function Clone: TPoseData;
  end;

  TAIPoseBuffer = class
  private
    FLock: TCriticalSection;
    FPoseHistory: TList<TPoseData>;
    FMaxHistorySize: Integer;
  public
    constructor Create(AMaxHistorySize: Integer = 100);
    destructor Destroy; override;
    procedure UpdatePose(ATimestamp: Int64; APoseData: TJSONArray);
    procedure RollbackToTimestamp(ATimestamp: Int64);
    function GetCurrentPose: TPoseData;
    function GetPoseCount: Integer;
    procedure Clear;
  end;

implementation

{ TPoseData }

function TPoseData.Clone: TPoseData;
begin
  Result.Timestamp := Self.Timestamp;
  if Assigned(Self.PoseArray) then
    Result.PoseArray := TJSONArray(Self.PoseArray.Clone)
  else
    Result.PoseArray := nil;
end;

{ TAIPoseBuffer }

constructor TAIPoseBuffer.Create(AMaxHistorySize: Integer);
begin
  inherited Create;
  FLock := TCriticalSection.Create;
  FPoseHistory := TList<TPoseData>.Create;
  FMaxHistorySize := AMaxHistorySize;
end;

destructor TAIPoseBuffer.Destroy;
var
  I: Integer;
begin
  FLock.Enter;
  try
    // Free all JSON arrays
    for I := 0 to FPoseHistory.Count - 1 do
    begin
      if Assigned(FPoseHistory[I].PoseArray) then
        FPoseHistory[I].PoseArray.Free;
    end;
    FPoseHistory.Free;
  finally
    FLock.Leave;
  end;
  FLock.Free;
  inherited;
end;

procedure TAIPoseBuffer.UpdatePose(ATimestamp: Int64; APoseData: TJSONArray);
var
  PoseData: TPoseData;
begin
  FLock.Enter;
  try
    PoseData.Timestamp := ATimestamp;
    PoseData.PoseArray := TJSONArray(APoseData.Clone);
    
    FPoseHistory.Add(PoseData);
    
    // Trim history if it exceeds max size
    while FPoseHistory.Count > FMaxHistorySize do
    begin
      if Assigned(FPoseHistory[0].PoseArray) then
        FPoseHistory[0].PoseArray.Free;
      FPoseHistory.Delete(0);
    end;
  finally
    FLock.Leave;
  end;
end;

procedure TAIPoseBuffer.RollbackToTimestamp(ATimestamp: Int64);
var
  I: Integer;
begin
  FLock.Enter;
  try
    // Remove all poses after the specified timestamp
    for I := FPoseHistory.Count - 1 downto 0 do
    begin
      if FPoseHistory[I].Timestamp > ATimestamp then
      begin
        if Assigned(FPoseHistory[I].PoseArray) then
          FPoseHistory[I].PoseArray.Free;
        FPoseHistory.Delete(I);
      end;
    end;
  finally
    FLock.Leave;
  end;
end;

function TAIPoseBuffer.GetCurrentPose: TPoseData;
begin
  FLock.Enter;
  try
    if FPoseHistory.Count > 0 then
      Result := FPoseHistory[FPoseHistory.Count - 1].Clone
    else
    begin
      Result.Timestamp := 0;
      Result.PoseArray := nil;
    end;
  finally
    FLock.Leave;
  end;
end;

function TAIPoseBuffer.GetPoseCount: Integer;
begin
  FLock.Enter;
  try
    Result := FPoseHistory.Count;
  finally
    FLock.Leave;
  end;
end;

procedure TAIPoseBuffer.Clear;
var
  I: Integer;
begin
  FLock.Enter;
  try
    for I := 0 to FPoseHistory.Count - 1 do
    begin
      if Assigned(FPoseHistory[I].PoseArray) then
        FPoseHistory[I].PoseArray.Free;
    end;
    FPoseHistory.Clear;
  finally
    FLock.Leave;
  end;
end;

end.
