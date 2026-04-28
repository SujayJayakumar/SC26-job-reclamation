$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$env:PYTHONHASHSEED = "42"

# Previous Linux/Swapnil-machine invocation kept for future switching:
# python3 -m src.hpc_sim.pipeline --config "$RootDir/config/default.json" --phase all --scenario all

$pythonCmd = if ($env:PYTHON_CMD) {
    $env:PYTHON_CMD
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    "python3"
} else {
    throw "No Python interpreter found. Set PYTHON_CMD or install python3."
}

& $pythonCmd -m src.hpc_sim.pipeline --config (Join-Path $RootDir "config/default.json") --phase all --scenario all @args
