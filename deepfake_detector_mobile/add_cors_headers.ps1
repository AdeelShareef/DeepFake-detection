# Add CORS Headers to upload_handler.php
# This script will backup your original file and add CORS headers

$phpFile = "D:\xampp\htdocs\deepshield\upload_handler.php"
$backupFile = "D:\xampp\htdocs\deepshield\upload_handler.php.backup"

# Create backup
Write-Host "Creating backup..." -ForegroundColor Yellow
Copy-Item $phpFile $backupFile -Force
Write-Host "✓ Backup created: $backupFile" -ForegroundColor Green

# Read current content
$content = Get-Content $phpFile -Raw

# Remove existing <?php tag
$content = $content -replace '^\s*<\?php\s*', ''

# CORS headers to add
$corsHeaders = @'
<?php
// CORS Headers - Allow Flutter web app requests
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, GET, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, Accept, Authorization');
header('Access-Control-Max-Age: 86400');

// Handle preflight OPTIONS request
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

'@

# Combine CORS headers with original content
$newContent = $corsHeaders + $content

# Write new content
Set-Content $phpFile -Value $newContent -NoNewline

Write-Host ""
Write-Host "✓ CORS headers added successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Restart Apache in XAMPP Control Panel" -ForegroundColor White
Write-Host "2. Hot restart your Flutter app (press R in terminal)" -ForegroundColor White
Write-Host ""
Write-Host "If something goes wrong, restore from backup:" -ForegroundColor Yellow
Write-Host "   Copy-Item '$backupFile' '$phpFile' -Force" -ForegroundColor Gray
