library DelphiBridge;

{ Important note about DLL memory management: ShareMem must be the
  first unit in your library's USES clause AND your project's (select
  Project-View Source) USES clause if your DLL exports any procedures or
  functions that pass strings as parameters or function results. This
  applies to all strings passed to and from your DLL--even those that
  are nested in records and classes. ShareMem is the interface unit to
  the BORLNDMM.DLL shared memory manager, which must be deployed along
  with your DLL. To avoid using BORLNDMM.DLL, pass string information
  using PChar or ShortString parameters. }

uses
  System.SysUtils,
  System.Classes,
  AudioStreamer in 'Source\AudioStreamer.pas',
  CommandRouter in 'Source\CommandRouter.pas',
  PoseBuffer in 'Source\PoseBuffer.pas',
  CommandBuffer in 'Source\CommandBuffer.pas',
  ExportedFunctions in 'Source\ExportedFunctions.pas',
  FuzzyUI in 'Source\FuzzyUI.pas' {FuzzyForm};

{$R *.res}

var
  AudioStreamer: TAudioStreamer;
  CommandRouter: TCommandRouter;
  FuzzyUIForm: TFuzzyForm;

// Library initialization and cleanup functions
function DB_Initialize(AAudioHost: PChar; AAudioPort: Integer; 
                        ACommandPort: Integer; ASidecarHost: PChar; 
                        ASidecarPort: Integer): Integer; cdecl;
begin
  Result := 0;
  try
    // Initialize Audio Streamer
    AudioStreamer := TAudioStreamer.Create(string(AAudioHost), AAudioPort);
    
    // Initialize Command Router
    CommandRouter := TCommandRouter.Create(ACommandPort);
    
    // Initialize Fuzzy UI
    Application.Initialize;
    Application.CreateForm(TFuzzyForm, FuzzyUIForm);
    FuzzyUIForm.Initialize(string(ASidecarHost), ASidecarPort);
    
    // Wire up command received callback to UI
    CommandRouter.OnCommandReceived := 
      procedure(AGroup, AVerb, AIntent: string)
      begin
        TThread.Synchronize(nil,
          procedure
          begin
            FuzzyUIForm.ShowIntent(AGroup, AVerb, AIntent, 0);
          end);
      end;
    
    Result := 1; // Success
  except
    on E: Exception do
      Result := 0; // Failure
  end;
end;

function DB_Start: Integer; cdecl;
begin
  Result := 0;
  try
    if Assigned(AudioStreamer) then
      AudioStreamer.Start;
      
    if Assigned(CommandRouter) then
      CommandRouter.Start;
      
    Result := 1; // Success
  except
    on E: Exception do
      Result := 0; // Failure
  end;
end;

function DB_Stop: Integer; cdecl;
begin
  Result := 0;
  try
    if Assigned(AudioStreamer) then
      AudioStreamer.Stop;
      
    if Assigned(CommandRouter) then
      CommandRouter.Stop;
      
    Result := 1; // Success
  except
    on E: Exception do
      Result := 0; // Failure
  end;
end;

procedure DB_Shutdown; cdecl;
begin
  try
    if Assigned(AudioStreamer) then
    begin
      AudioStreamer.Stop;
      FreeAndNil(AudioStreamer);
    end;
    
    if Assigned(CommandRouter) then
    begin
      CommandRouter.Stop;
      FreeAndNil(CommandRouter);
    end;
    
    if Assigned(FuzzyUIForm) then
      FuzzyUIForm.Close;
  except
    // Ignore errors during shutdown
  end;
end;

function DB_RouteCommand(AJson: PChar): Integer; cdecl;
begin
  Result := 0;
  try
    if Assigned(CommandRouter) then
    begin
      if CommandRouter.RouteCommand(string(AJson)) then
        Result := 1
      else
        Result := 0;
    end;
  except
    on E: Exception do
      Result := -1; // Error
  end;
end;

function DB_GetPoseCount: Integer; cdecl;
begin
  Result := 0;
  try
    if Assigned(CommandRouter) and Assigned(CommandRouter.PoseBuffer) then
      Result := CommandRouter.PoseBuffer.GetPoseCount;
  except
    on E: Exception do
      Result := -1;
  end;
end;

function DB_GetCommandCount: Integer; cdecl;
begin
  Result := 0;
  try
    if Assigned(CommandRouter) and Assigned(CommandRouter.CommandBuffer) then
      Result := CommandRouter.CommandBuffer.GetCommandCount;
  except
    on E: Exception do
      Result := -1;
  end;
end;

exports
  DB_Initialize,
  DB_Start,
  DB_Stop,
  DB_Shutdown,
  DB_RouteCommand,
  DB_GetPoseCount,
  DB_GetCommandCount,
  FF_HandleCameraCommand,
  FF_HandleObjectCommand;

begin
end.
