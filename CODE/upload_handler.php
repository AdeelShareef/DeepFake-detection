<?php
session_start();
include('db_connection.php');

if(isset($_FILES['mediaFile'])) {
    $target_dir = "uploads/";
    $filename = basename($_FILES["mediaFile"]["name"]);
    $target_file = $target_dir . $filename;
    
    // Save file
    if(move_uploaded_file($_FILES["mediaFile"]["tmp_name"], $target_file)) {
        
        // Execute Python Script
        // NOTE: Adjust 'python' to 'python3' if on Mac/Linux.
        // If using XAMPP on Windows, you might need the full path like:
        // C:\\Users\\Name\\AppData\\Local\\Programs\\Python\\Python39\\python.exe
        $command = "python python/predict.py " . escapeshellarg($target_file);
        $output = shell_exec($command);
        
        // Decode JSON from Python
        $resultData = json_decode($output, true);
        
        // Fallback if python fails (for testing)
        if ($resultData == null) {
            $resultData = ["result" => "ERROR", "confidence" => "0.00"];
        }

        $pred = $resultData['result'];
        $conf = $resultData['confidence'];

        // Save to Database
        $uid = $_SESSION['user_id'];
        $stmt = $conn->prepare("INSERT INTO history (user_id, filename, prediction, confidence) VALUES (?, ?, ?, ?)");
        $stmt->bind_param("isss", $uid, $filename, $pred, $conf);
        $stmt->execute();

        header("Location: dashboard.php?result=$pred&conf=$conf");
    } else {
        echo "Sorry, there was an error uploading your file.";
    }
}
?>