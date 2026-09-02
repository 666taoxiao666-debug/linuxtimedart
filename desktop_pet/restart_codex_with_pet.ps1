$ErrorActionPreference = 'Stop'

$statePath = 'C:\Users\XT\.codex\.codex-global-state.json'
$backupPath = 'C:\Users\XT\.codex\.codex-global-state.before-pet-restart.json'
$logPath = 'C:\Users\XT\.codex\pets\girlfriend\restart.log'
$codexExe = 'C:\Program Files\WindowsApps\OpenAI.Codex_26.825.6671.0_x64__2p2nqsd0c76g0\app\ChatGPT.exe'

"$(Get-Date -Format o) restart task started" |
    Set-Content -LiteralPath $logPath -Encoding utf8

Start-Sleep -Seconds 12

try {
    $codexProcesses = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'ChatGPT.exe' -and $_.ExecutablePath -eq $codexExe
    }
    foreach ($codexProcess in $codexProcesses) {
        Stop-Process -Id $codexProcess.ProcessId -Force -ErrorAction SilentlyContinue
    }

    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        $stillRunning = Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq 'ChatGPT.exe' -and $_.ExecutablePath -eq $codexExe
        }
        if (-not $stillRunning) { break }
        Start-Sleep -Milliseconds 500
    }

    Copy-Item -LiteralPath $statePath -Destination $backupPath -Force
    $state = Get-Content -Raw -Encoding utf8 $statePath | ConvertFrom-Json -AsHashtable
    $state['selected-avatar-id'] = 'custom:girlfriend'
    $state['avatar-overlay-mascot-width-px'] = 160
    $state['avatar-overlay-pet-visible'] = $true
    $state['electron-avatar-overlay-open'] = $true
    $state | ConvertTo-Json -Depth 100 -Compress | Set-Content -LiteralPath $statePath -Encoding utf8 -NoNewline

    "$(Get-Date -Format o) configured selected-avatar-id=custom:girlfriend size=160 visible=true open=true" |
        Add-Content -LiteralPath $logPath -Encoding utf8

    Start-Process -FilePath $codexExe
    "$(Get-Date -Format o) Codex relaunched" |
        Add-Content -LiteralPath $logPath -Encoding utf8
} catch {
    "$(Get-Date -Format o) ERROR: $($_.Exception.Message)" |
        Add-Content -LiteralPath $logPath -Encoding utf8
    throw
}
