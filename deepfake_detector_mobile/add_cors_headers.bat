@echo off
REM Backup original file
echo Creating backup...
copy "D:\xampp\htdocs\deepshield\upload_handler.php" "D:\xampp\htdocs\deepshield\upload_handler.php.backup"

REM Create new file with CORS headers
echo Adding CORS headers...
(
echo ^<?php
echo // CORS Headers - Allow Flutter web app requests
echo header('Access-Control-Allow-Origin: *'^);
echo header('Access-Control-Allow-Methods: POST, GET, OPTIONS'^);
echo header('Access-Control-Allow-Headers: Content-Type, Accept, Authorization'^);
echo header('Access-Control-Max-Age: 86400'^);
echo.
echo // Handle preflight OPTIONS request
echo if ^($_SERVER['REQUEST_METHOD'] === 'OPTIONS'^) {
echo     http_response_code^(200^);
echo     exit^(^);
echo }
echo.
type "D:\xampp\htdocs\deepshield\upload_handler.php.backup" ^| findstr /v "^<?php"
) > "D:\xampp\htdocs\deepshield\upload_handler_new.php"

REM Replace old file with new one
move /Y "D:\xampp\htdocs\deepshield\upload_handler_new.php" "D:\xampp\htdocs\deepshield\upload_handler.php"

echo.
echo ✓ CORS headers added successfully!
echo ✓ Backup saved as: upload_handler.php.backup
echo.
echo Next steps:
echo 1. Restart Apache in XAMPP Control Panel
echo 2. Hot restart your Flutter app (press R in terminal)
echo.
pause
