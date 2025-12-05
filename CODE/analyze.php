<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Analysis Result | DeepShield</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body { background-color: #f0f2f5; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .result-card { margin-top: 80px; border-radius: 20px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .result-header { padding: 30px; text-align: center; color: white; }
        .bg-real { background: linear-gradient(135deg, #42e695 0%, #3bb2b8 100%); }
        .bg-fake { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); }
        .bg-error { background: #6c757d; }
    </style>
</head>
<body>

<div class="container">
    <div class="row justify-content-center">
        <div class="col-lg-6">
            <div class="card result-card">
                <?php
                $uploadDir = 'uploads/';
                // Create uploads folder if it doesn't exist
                if (!is_dir($uploadDir)) mkdir($uploadDir, 0777, true);

                $uploadFile = $uploadDir . basename($_FILES['media_file']['name']);
                $fileType = strtolower(pathinfo($uploadFile, PATHINFO_EXTENSION));
                $headerClass = "bg-error";
                $icon = "fa-exclamation-triangle";
                $title = "Processing Error";
                $message = "An error occurred.";

                // 1. Move Uploaded File
                if (move_uploaded_file($_FILES['media_file']['tmp_name'], $uploadFile)) {
                    
                    // 2. Call Python Script
                    // Note: We use 2>&1 to capture error messages from Python if any
                    $command = escapeshellcmd("python predict.py " . escapeshellarg($uploadFile));
                    $output = shell_exec($command . " 2>&1");
                    
                    // 3. Parse Output
                    $output = trim($output);
                    
                    if (strpos($output, 'FAKE') !== false) {
                        $headerClass = "bg-fake";
                        $icon = "fa-user-secret";
                        $title = "DEEPFAKE DETECTED";
                        $message = "This media shows strong signs of AI manipulation.";
                    } elseif (strpos($output, 'REAL') !== false) {
                        $headerClass = "bg-real";
                        $icon = "fa-check-circle";
                        $title = "AUTHENTIC MEDIA";
                        $message = "No significant manipulation detected.";
                    } else {
                        // Pass through Python error for debugging
                        $message = "System Output: " . htmlspecialchars($output);
                    }

                } else {
                    $message = "Failed to upload file. Check folder permissions.";
                }
                ?>

                <!-- Display Result -->
                <div class="result-header <?php echo $headerClass; ?>">
                    <i class="fas <?php echo $icon; ?> fa-5x mb-3"></i>
                    <h2><?php echo $title; ?></h2>
                </div>
                <div class="card-body p-5 text-center">
                    <p class="lead text-muted"><?php echo $message; ?></p>
                    <hr>
                    <p class="small text-secondary">Analyzed File: <?php echo basename($_FILES['media_file']['name']); ?></p>
                    
                    <a href="index.html" class="btn btn-outline-dark btn-lg mt-3 rounded-pill px-5">
                        <i class="fas fa-redo me-2"></i> Analyze Another
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>

</body>
</html>