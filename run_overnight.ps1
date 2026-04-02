# FB-CycleGAN Overnight Experiment Suite
# Run all evaluation experiments while you sleep.
#
# Usage: powershell -ExecutionPolicy Bypass -File run_overnight.ps1

$ErrorActionPreference = "Continue"
$env:MPLBACKEND = "Agg"
$env:PYTHONUNBUFFERED = "1"
$python = ".venv\Scripts\python.exe"
$startTime = Get-Date

Write-Host "=============================================="
Write-Host " FB-CycleGAN Overnight Experiments"
Write-Host " Started: $startTime"
Write-Host "=============================================="
Write-Host ""

# Create output directories
New-Item -ItemType Directory -Force -Path "outputs\plots" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\metrics" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\logs" | Out-Null

# Log file for this run
$logFile = "outputs\logs\overnight_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Run-Experiment {
    param([string]$Name, [string]$Command)

    $expStart = Get-Date
    Write-Host "----------------------------------------------"
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] STARTING: $Name"
    Write-Host "----------------------------------------------"

    # Run and capture output
    $output = & $python -m $Command 2>&1 | Tee-Object -Append -FilePath $logFile
    $exitCode = $LASTEXITCODE

    $elapsed = (Get-Date) - $expStart
    if ($exitCode -eq 0) {
        Write-Host "  DONE in $($elapsed.ToString('hh\:mm\:ss')) [OK]"
    } else {
        Write-Host "  FAILED in $($elapsed.ToString('hh\:mm\:ss')) [EXIT CODE: $exitCode]"
    }
    Write-Host ""

    # Add to log
    Add-Content -Path $logFile -Value ""
    Add-Content -Path $logFile -Value "=== $Name completed in $($elapsed.ToString('hh\:mm\:ss')) (exit=$exitCode) ==="
    Add-Content -Path $logFile -Value ""
}

# ============================================
# Experiment 1: Forensic FFT Audit (all models)
# ============================================
Write-Host ""
Write-Host "###############################################"
Write-Host "# PHASE 1: Forensic FFT Audits"
Write-Host "###############################################"
Write-Host ""

Run-Experiment "FFT Audit: Baseline" "scripts.forensic_audit --checkpoint outputs/checkpoints/baseline/final.pt --n-samples 100 --name baseline"
Run-Experiment "FFT Audit: FB s=0.5" "scripts.forensic_audit --checkpoint outputs/checkpoints/fb/fb_sigma0.5.pt --n-samples 100 --name fb_sigma0.5"
Run-Experiment "FFT Audit: FB s=1.0" "scripts.forensic_audit --checkpoint outputs/checkpoints/fb/fb_sigma1.0.pt --n-samples 100 --name fb_sigma1.0"
Run-Experiment "FFT Audit: FB s=1.5" "scripts.forensic_audit --checkpoint outputs/checkpoints/fb/fb_sigma1.5.pt --n-samples 100 --name fb_sigma1.5"
Run-Experiment "FFT Audit: FB s=2.0" "scripts.forensic_audit --checkpoint outputs/checkpoints/fb/fb_sigma2.pt --n-samples 100 --name fb_sigma2.0"

# ============================================
# Experiment 2: Pareto Analysis
# ============================================
Write-Host ""
Write-Host "###############################################"
Write-Host "# PHASE 2: Pareto Analysis (perturbation sweep)"
Write-Host "###############################################"
Write-Host ""

Run-Experiment "Pareto Analysis" "scripts.pareto_analysis"

# ============================================
# Experiment 3: FID / SSIM Evaluation
# ============================================
Write-Host ""
Write-Host "###############################################"
Write-Host "# PHASE 3: FID / SSIM Quality Metrics"
Write-Host "###############################################"
Write-Host ""

Run-Experiment "FID / SSIM Evaluation" "scripts.eval_fid_ssim"

# ============================================
# Experiment 4: Classifier Leakage Test
# ============================================
Write-Host ""
Write-Host "###############################################"
Write-Host "# PHASE 4: Classifier Leakage Test"
Write-Host "###############################################"
Write-Host ""

Run-Experiment "Classifier Leakage Test" "scripts.classifier_leakage"

# ============================================
# Summary
# ============================================
$totalElapsed = (Get-Date) - $startTime

Write-Host ""
Write-Host "=============================================="
Write-Host " ALL EXPERIMENTS COMPLETE"
Write-Host " Total time: $($totalElapsed.ToString('hh\:mm\:ss'))"
Write-Host " Log file: $logFile"
Write-Host "=============================================="
Write-Host ""
Write-Host "Generated outputs:"
Write-Host "  outputs/plots/forensic_audit_baseline.png"
Write-Host "  outputs/plots/forensic_audit_fb_sigma*.png"
Write-Host "  outputs/plots/pareto_analysis.png"
Write-Host "  outputs/plots/fid_ssim_comparison.png"
Write-Host "  outputs/plots/classifier_leakage.png"
Write-Host "  outputs/metrics/fid_ssim_results.json"
Write-Host "  $logFile"
Write-Host ""
Write-Host "Review the plots in outputs/plots/ when you wake up!"
