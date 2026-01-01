unit AudioStreamer;

interface

uses
  System.SysUtils, System.Classes, System.Threading,
  FMX.Media, System.Net.Socket;

type
  TAudioStreamer = class
  private
    FAudioCaptureDevice: TAudioCaptureDevice;
    FTCPClient: TSocket;
    FStreamThread: ITask;
    FActive: Boolean;
    FHost: string;
    FPort: Integer;
    procedure OnAudioCaptureSample(Sender: TObject; const ATime: TMediaTime);
    procedure ConnectToServer;
    procedure DisconnectFromServer;
  public
    constructor Create(const AHost: string = 'localhost'; APort: Integer = 9000);
    destructor Destroy; override;
    procedure Start;
    procedure Stop;
    property Active: Boolean read FActive;
  end;

implementation

{ TAudioStreamer }

constructor TAudioStreamer.Create(const AHost: string; APort: Integer);
begin
  inherited Create;
  FHost := AHost;
  FPort := APort;
  FActive := False;
  
  // Initialize audio capture device
  FAudioCaptureDevice := TAudioCaptureDevice.Create;
  FAudioCaptureDevice.OnSampleBufferReady := OnAudioCaptureSample;
end;

destructor TAudioStreamer.Destroy;
begin
  Stop;
  FAudioCaptureDevice.Free;
  inherited;
end;

procedure TAudioStreamer.ConnectToServer;
begin
  try
    FTCPClient := TSocket.Create(TSocketType.TCP);
    FTCPClient.Connect('', FHost, '', FPort);
  except
    on E: Exception do
      raise Exception.CreateFmt('Failed to connect to %s:%d - %s', [FHost, FPort, E.Message]);
  end;
end;

procedure TAudioStreamer.DisconnectFromServer;
begin
  if Assigned(FTCPClient) then
  begin
    FTCPClient.Close;
    FTCPClient := nil;
  end;
end;

procedure TAudioStreamer.OnAudioCaptureSample(Sender: TObject; const ATime: TMediaTime);
var
  Buffer: TArray<Byte>;
  SampleData: Pointer;
  SampleSize: Integer;
begin
  if not FActive then
    Exit;
    
  // Get raw PCM data from the audio capture device
  if FAudioCaptureDevice.SampleReady then
  begin
    SampleSize := FAudioCaptureDevice.SampleSize;
    if SampleSize > 0 then
    begin
      SetLength(Buffer, SampleSize);
      SampleData := FAudioCaptureDevice.SampleBuffer;
      Move(SampleData^, Buffer[0], SampleSize);
      
      // Send PCM data over socket
      if Assigned(FTCPClient) then
      begin
        try
          FTCPClient.SendData(Buffer);
        except
          on E: Exception do
          begin
            // Log error but don't stop streaming
            // Could implement reconnection logic here
          end;
        end;
      end;
    end;
  end;
end;

procedure TAudioStreamer.Start;
begin
  if FActive then
    Exit;
    
  ConnectToServer;
  
  // Start audio capture
  FAudioCaptureDevice.StartCapture;
  FActive := True;
end;

procedure TAudioStreamer.Stop;
begin
  if not FActive then
    Exit;
    
  FActive := False;
  
  // Stop audio capture
  if Assigned(FAudioCaptureDevice) then
    FAudioCaptureDevice.StopCapture;
    
  DisconnectFromServer;
end;

end.
