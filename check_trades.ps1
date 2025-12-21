# Quick PowerShell script to check trade dates from CSV files
# Run this in PowerShell: .\check_trades.ps1

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "CHECKING TRADE DATES" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if any trade CSV files exist
$tradeFiles = Get-ChildItem -Path "report_html" -Recurse -Filter "trades_*.csv" -ErrorAction SilentlyContinue

if ($tradeFiles.Count -eq 0) {
    Write-Host "No trade CSV files found in report_html/" -ForegroundColor Yellow
    Write-Host "You may need to run the full simulation first.`n" -ForegroundColor Yellow
    exit
}

Write-Host "Found $($tradeFiles.Count) trade CSV files`n" -ForegroundColor Green

# Analyze each file
$allFirstDates = @()
$jan2025Count = 0
$totalTrades = 0

foreach ($file in $tradeFiles) {
    try {
        $csv = Import-Csv -Path $file.FullName -Delimiter ";"

        if ($csv.Count -gt 0 -and $csv[0].PSObject.Properties.Name -contains "EntryTime") {
            $dates = $csv | ForEach-Object { [DateTime]::Parse($_.EntryTime) } | Sort-Object

            if ($dates.Count -gt 0) {
                $firstDate = $dates[0]
                $lastDate = $dates[-1]
                $jan2025 = ($dates | Where-Object { $_.Year -eq 2025 -and $_.Month -eq 1 }).Count

                $allFirstDates += [PSCustomObject]@{
                    File = $file.Name
                    FirstTrade = $firstDate
                    LastTrade = $lastDate
                    TotalTrades = $dates.Count
                    Jan2025 = $jan2025
                }

                $totalTrades += $dates.Count
                $jan2025Count += $jan2025
            }
        }
    }
    catch {
        Write-Host "Error reading $($file.Name): $_" -ForegroundColor Red
    }
}

if ($allFirstDates.Count -eq 0) {
    Write-Host "No valid trade data found`n" -ForegroundColor Yellow
    exit
}

# Sort by first trade date
$allFirstDates = $allFirstDates | Sort-Object FirstTrade

# Show summary
Write-Host "EARLIEST TRADES:" -ForegroundColor Cyan
Write-Host "-" * 80
$allFirstDates | Select-Object -First 10 | ForEach-Object {
    $dateStr = $_.FirstTrade.ToString("yyyy-MM-dd HH:mm")
    Write-Host ("  {0,-50} {1}" -f $_.File.Substring(0, [Math]::Min(50, $_.File.Length)), $dateStr)
}

Write-Host "`n" + ("-" * 80)
Write-Host ("TOTAL: {0} trades across all strategies" -f $totalTrades) -ForegroundColor Green
Write-Host ("January 2025: {0} trades" -f $jan2025Count) -ForegroundColor $(if ($jan2025Count -eq 0) { "Red" } else { "Green" })
Write-Host ("-" * 80)

# Key finding
$veryFirstTrade = ($allFirstDates | Sort-Object FirstTrade | Select-Object -First 1).FirstTrade
Write-Host "`nFIRST TRADE ACROSS ALL STRATEGIES: $($veryFirstTrade.ToString('yyyy-MM-dd HH:mm'))" -ForegroundColor Yellow

if ($jan2025Count -eq 0) {
    Write-Host "`n⚠ WARNING: No trades in January 2025!" -ForegroundColor Red
    Write-Host "  This explains why --start 2025-01-01 --end 2025-01-31 produced 0 trades.`n" -ForegroundColor Red
}

Write-Host ""
