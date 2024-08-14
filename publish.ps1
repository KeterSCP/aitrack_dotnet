param (
    [string]$runtime,
    [string]$option
)

$initialDir = Get-Location

if ($runtime -eq "win-x64") {
    Push-Location -Path "AITrackDotnet" -PassThru
    if ($option -eq "cuda") {
        dotnet publish -c Release -r win-x64 -p:DefineConstants=USE_CUDA -o "$initialDir\artifacts\publish_cuda"
    } else {
        dotnet publish -c Release -r win-x64 -o "$initialDir\artifacts\publish_cpu"
    }
} else {
    Write-Host "Usage: .\publish.ps1 <runtime> [cuda]"
    Write-Host "Example: .\publish.ps1 win-x64 [cuda]"

    exit 0
}

Pop-Location -PassThru