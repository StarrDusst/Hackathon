$ErrorActionPreference = 'Stop'

# Reproducible Windows setup for local inference and the upload demo.
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    py -3 -m venv .venv
}
$Python = '.venv\Scripts\python.exe'
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt
# PyTorch 2.11 requires setuptools below version 82.
& $Python -m pip install --upgrade 'setuptools<82'
# Replace PyPI's CPU-only Windows wheels with CUDA 12.8 wheels for NVIDIA GPUs.
$CudaWheelDir = Join-Path (Get-Location).Path '.cuda-wheels'
$CudaTemp = Join-Path (Get-Location).Path '.cuda-tmp'
New-Item -ItemType Directory -Force $CudaWheelDir, $CudaTemp | Out-Null
$OldTemp = $env:TEMP; $OldTmp = $env:TMP
try {
    $env:TEMP = $CudaTemp; $env:TMP = $CudaTemp
    & $Python -m pip download --only-binary=:all: --no-deps --dest $CudaWheelDir --timeout 180 --retries 20 --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchvision==0.26.0+cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA wheel download failed; rerun setup.ps1 to retry.' }
    & $Python -m pip install --upgrade --no-deps --no-index --find-links $CudaWheelDir torch==2.11.0+cu128 torchvision==0.26.0+cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA wheel installation failed.' }
} finally {
    $env:TEMP = $OldTemp; $env:TMP = $OldTmp
}
Remove-Item -LiteralPath $CudaWheelDir, $CudaTemp -Recurse -Force -ErrorAction SilentlyContinue
& $Python weights\download.py
Write-Host ''
Write-Host 'Setup complete. Activate with .\.venv\Scripts\Activate.ps1'
Write-Host 'Run samples: python run_submission.py --videos ..\samples --out predictions_samples.json --team "Qwen 3.8"'
Write-Host 'Run demo:    streamlit run app.py'
