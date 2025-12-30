program SimpleExample;

{$APPTYPE CONSOLE}

uses
  System.SysUtils,
  System.JSON,
  WinApi.Windows;

type
  TDB_Initialize = function(AAudioHost: PChar; AAudioPort: Integer; 
                            ACommandPort: Integer; ASidecarHost: PChar; 
                            ASidecarPort: Integer): Integer; cdecl;
  TDB_Start = function: Integer; cdecl;
  TDB_Stop = function: Integer; cdecl;
  TDB_Shutdown = procedure; cdecl;
  TDB_RouteCommand = function(AJson: PChar): Integer; cdecl;
  TDB_GetPoseCount = function: Integer; cdecl;
  TDB_GetCommandCount = function: Integer; cdecl;

var
  LibHandle: HMODULE;
  DB_Initialize: TDB_Initialize;
  DB_Start: TDB_Start;
  DB_Stop: TDB_Stop;
  DB_Shutdown: TDB_Shutdown;
  DB_RouteCommand: TDB_RouteCommand;
  DB_GetPoseCount: TDB_GetPoseCount;
  DB_GetCommandCount: TDB_GetCommandCount;

procedure LoadLibrary;
begin
  LibHandle := WinApi.Windows.LoadLibrary('DelphiBridge.dll');
  if LibHandle = 0 then
    raise Exception.Create('Failed to load DelphiBridge.dll');
    
  @DB_Initialize := GetProcAddress(LibHandle, 'DB_Initialize');
  @DB_Start := GetProcAddress(LibHandle, 'DB_Start');
  @DB_Stop := GetProcAddress(LibHandle, 'DB_Stop');
  @DB_Shutdown := GetProcAddress(LibHandle, 'DB_Shutdown');
  @DB_RouteCommand := GetProcAddress(LibHandle, 'DB_RouteCommand');
  @DB_GetPoseCount := GetProcAddress(LibHandle, 'DB_GetPoseCount');
  @DB_GetCommandCount := GetProcAddress(LibHandle, 'DB_GetCommandCount');
  
  if not Assigned(DB_Initialize) then
    raise Exception.Create('Failed to get function pointers');
end;

procedure UnloadLibrary;
begin
  if LibHandle <> 0 then
  begin
    DB_Shutdown;
    FreeLibrary(LibHandle);
    LibHandle := 0;
  end;
end;

procedure SendCameraCommand(AVerb, ASubject: string; AAngle: Double; ATimestamp: Int64);
var
  JsonObj: TJSONObject;
  ParamsObj: TJSONObject;
  JsonStr: string;
begin
  JsonObj := TJSONObject.Create;
  try
    JsonObj.AddPair('timestamp', TJSONNumber.Create(ATimestamp));
    JsonObj.AddPair('group', 'CAMERA_CONTROL');
    JsonObj.AddPair('verb', AVerb);
    JsonObj.AddPair('subject', ASubject);
    
    ParamsObj := TJSONObject.Create;
    ParamsObj.AddPair('angle', TJSONNumber.Create(AAngle));
    JsonObj.AddPair('params', ParamsObj);
    
    JsonStr := JsonObj.ToJSON;
    WriteLn('Sending: ', JsonStr);
    
    if DB_RouteCommand(PChar(JsonStr)) = 1 then
      WriteLn('Command sent successfully')
    else
      WriteLn('Command failed');
  finally
    JsonObj.Free;
  end;
end;

procedure SendPoseCommand(APoseData: TJSONArray; ATimestamp: Int64);
var
  JsonObj: TJSONObject;
  ParamsObj: TJSONObject;
  JsonStr: string;
begin
  JsonObj := TJSONObject.Create;
  try
    JsonObj.AddPair('timestamp', TJSONNumber.Create(ATimestamp));
    JsonObj.AddPair('group', 'ACTOR_POSE');
    JsonObj.AddPair('verb', 'UPDATE');
    JsonObj.AddPair('subject', 'character_1');
    
    ParamsObj := TJSONObject.Create;
    ParamsObj.AddPair('pose_data', APoseData);
    JsonObj.AddPair('params', ParamsObj);
    
    JsonStr := JsonObj.ToJSON;
    WriteLn('Sending: ', JsonStr);
    
    if DB_RouteCommand(PChar(JsonStr)) = 1 then
      WriteLn('Pose command sent successfully')
    else
      WriteLn('Pose command failed');
  finally
    JsonObj.Free;
  end;
end;

var
  PoseArray: TJSONArray;
  Timestamp: Int64;

begin
  try
    WriteLn('Delphi Bridge Example');
    WriteLn('====================');
    WriteLn;
    
    // Load the library
    WriteLn('Loading DelphiBridge.dll...');
    LoadLibrary;
    WriteLn('Library loaded successfully');
    WriteLn;
    
    // Initialize
    WriteLn('Initializing...');
    if DB_Initialize('localhost', 9000, 9001, 'localhost', 9002) = 1 then
      WriteLn('Initialized successfully')
    else
      raise Exception.Create('Initialization failed');
    WriteLn;
    
    // Start
    WriteLn('Starting...');
    if DB_Start = 1 then
      WriteLn('Started successfully')
    else
      raise Exception.Create('Start failed');
    WriteLn;
    
    // Send some commands
    WriteLn('Sending commands...');
    WriteLn;
    
    // Camera command at T=1000
    Timestamp := 1000;
    SendCameraCommand('PAN', 'main_camera', 25.0, Timestamp);
    WriteLn;
    Sleep(500);
    
    // Camera command at T=2000
    Timestamp := 2000;
    SendCameraCommand('TILT', 'main_camera', -10.0, Timestamp);
    WriteLn;
    Sleep(500);
    
    // Pose command at T=3000
    Timestamp := 3000;
    PoseArray := TJSONArray.Create;
    try
      PoseArray.Add(TJSONNumber.Create(1.0));
      PoseArray.Add(TJSONNumber.Create(2.0));
      PoseArray.Add(TJSONNumber.Create(3.0));
      SendPoseCommand(TJSONArray(PoseArray.Clone), Timestamp);
    finally
      PoseArray.Free;
    end;
    WriteLn;
    Sleep(500);
    
    // Demonstrate race condition handling - send command with earlier timestamp
    WriteLn('Demonstrating race condition handling...');
    Timestamp := 1500; // Between first two commands
    SendCameraCommand('ZOOM', 'main_camera', 2.0, Timestamp);
    WriteLn('(This should trigger rollback of commands after T=1500)');
    WriteLn;
    Sleep(500);
    
    // Check buffer status
    WriteLn('Buffer status:');
    WriteLn('  Pose count: ', DB_GetPoseCount);
    WriteLn('  Command count: ', DB_GetCommandCount);
    WriteLn;
    
    // Wait for user input
    WriteLn('Press Enter to stop...');
    ReadLn;
    
    // Stop
    WriteLn('Stopping...');
    if DB_Stop = 1 then
      WriteLn('Stopped successfully')
    else
      WriteLn('Stop failed');
    WriteLn;
    
    // Cleanup
    WriteLn('Cleaning up...');
    UnloadLibrary;
    WriteLn('Done');
    
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
