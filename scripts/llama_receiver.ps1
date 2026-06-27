param(
    [int]$port = 9000
)

$prefix = "http://127.0.0.1:$port/"
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add($prefix)
$listener.Start()
Write-Output "Listening on $prefix"
try {
    while ($true) {
        $context = $listener.GetContext()
        $request = $context.Request
        $response = $context.Response
        Write-Output "Received $($request.HttpMethod) $($request.Url.AbsolutePath)"
        Write-Output "Headers:";
        $request.Headers.GetEnumerator() | ForEach-Object { Write-Output "  $($_.Name): $($_.Value)" }
        $reader = New-Object System.IO.StreamReader($request.InputStream)
        $body = $reader.ReadToEnd()
        Write-Output "Body:";
        Write-Output $body
        $reader.Close()
        $buffer = [System.Text.Encoding]::UTF8.GetBytes('OK')
        $response.StatusCode = 200
        $response.ContentLength64 = $buffer.Length
        $response.OutputStream.Write($buffer, 0, $buffer.Length)
        $response.OutputStream.Close()
        [Console]::Out.Flush()
    }
} finally {
    $listener.Stop()
}
