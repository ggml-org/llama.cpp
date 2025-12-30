unit FuzzyUI;

interface

uses
  System.SysUtils, System.Types, System.UITypes, System.Classes, System.Variants,
  FMX.Types, FMX.Controls, FMX.Forms, FMX.Graphics, FMX.Dialogs, FMX.StdCtrls,
  FMX.Layouts, FMX.Objects, System.Net.Socket, System.JSON;

type
  TFuzzyForm = class(TForm)
    PanelMain: TPanel;
    LabelIntent: TLabel;
    ButtonUndo: TButton;
    RectangleBackground: TRectangle;
    procedure ButtonUndoClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private
    FSidecarHost: string;
    FSidecarPort: Integer;
    FCurrentTimestamp: Int64;
    procedure SendCancellationSignal;
  public
    procedure ShowIntent(const AGroup, AVerb, AIntentDescription: string; ATimestamp: Int64);
    procedure Initialize(const ASidecarHost: string; ASidecarPort: Integer);
  end;

var
  FuzzyForm: TFuzzyForm;

implementation

{$R *.fmx}

procedure TFuzzyForm.FormCreate(Sender: TObject);
begin
  // Set form properties for floating window
  Self.BorderStyle := TFmxFormBorderStyle.ToolWindow;
  Self.Position := TFormPosition.ScreenCenter;
  Self.Width := 400;
  Self.Height := 150;
  Self.Visible := False;
  
  // Style the background
  RectangleBackground.Fill.Color := TAlphaColorRec.Lightgray;
  RectangleBackground.Stroke.Color := TAlphaColorRec.Gray;
  RectangleBackground.Stroke.Thickness := 2;
  
  // Style the label
  LabelIntent.TextSettings.Font.Size := 14;
  LabelIntent.TextSettings.FontColor := TAlphaColorRec.Black;
  LabelIntent.Text := 'Waiting for command...';
  
  // Style the button
  ButtonUndo.Text := 'Undo';
  ButtonUndo.Enabled := False;
  
  FSidecarHost := 'localhost';
  FSidecarPort := 9002;
  FCurrentTimestamp := 0;
end;

procedure TFuzzyForm.Initialize(const ASidecarHost: string; ASidecarPort: Integer);
begin
  FSidecarHost := ASidecarHost;
  FSidecarPort := ASidecarPort;
end;

procedure TFuzzyForm.ShowIntent(const AGroup, AVerb, AIntentDescription: string; ATimestamp: Int64);
begin
  FCurrentTimestamp := ATimestamp;
  
  // Update label with recognized intent
  LabelIntent.Text := 'Recognized Intent: ' + AIntentDescription;
  
  // Enable undo button
  ButtonUndo.Enabled := True;
  
  // Show the form if hidden
  if not Self.Visible then
  begin
    Self.Show;
    Self.BringToFront;
  end;
  
  // Auto-hide after 5 seconds
  TThread.CreateAnonymousThread(
    procedure
    begin
      Sleep(5000);
      TThread.Synchronize(nil,
        procedure
        begin
          if Self.Visible then
          begin
            ButtonUndo.Enabled := False;
            Self.Hide;
          end;
        end);
    end).Start;
end;

procedure TFuzzyForm.ButtonUndoClick(Sender: TObject);
begin
  SendCancellationSignal;
  ButtonUndo.Enabled := False;
  LabelIntent.Text := 'Undo signal sent';
  
  // Hide form after a moment
  TThread.CreateAnonymousThread(
    procedure
    begin
      Sleep(1000);
      TThread.Synchronize(nil,
        procedure
        begin
          Self.Hide;
        end);
    end).Start;
end;

procedure TFuzzyForm.SendCancellationSignal;
var
  TCPClient: TSocket;
  CancelJson: TJSONObject;
  JsonStr: string;
  Data: TBytes;
begin
  try
    // Create cancellation JSON
    CancelJson := TJSONObject.Create;
    try
      CancelJson.AddPair('timestamp', TJSONNumber.Create(FCurrentTimestamp));
      CancelJson.AddPair('group', 'SYSTEM');
      CancelJson.AddPair('verb', 'CANCEL');
      CancelJson.AddPair('subject', 'last_command');
      CancelJson.AddPair('params', TJSONObject.Create);
      
      JsonStr := CancelJson.ToJSON + #10;
      
      // Send to Sidecar
      TCPClient := TSocket.Create(TSocketType.TCP);
      try
        TCPClient.Connect('', FSidecarHost, '', FSidecarPort);
        Data := TEncoding.UTF8.GetBytes(JsonStr);
        TCPClient.SendData(Data);
      finally
        TCPClient.Close;
      end;
    finally
      CancelJson.Free;
    end;
  except
    on E: Exception do
    begin
      // Log error
      LabelIntent.Text := 'Error sending undo: ' + E.Message;
    end;
  end;
end;

end.
