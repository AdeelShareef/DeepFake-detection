<?php
session_start();
include('db_connection.php');

if (!isset($_FILES['mediaFile'])) {
    echo json_encode(["error" => "No file uploaded"]);
    exit;
}

$uploadDir = __DIR__ . "/uploads/";
if (!file_exists($uploadDir)) mkdir($uploadDir, 0777, true);

$file = basename($_FILES["mediaFile"]["name"]);
$path = $uploadDir . $file;
move_uploaded_file($_FILES["mediaFile"]["tmp_name"], $path);

$python = "C:\\Program Files\\Python310\\python.exe";

$script = __DIR__ . "\\python\\predict.py";

$cmd = "\"$python\" \"$script\" \"$path\"";
$result = shell_exec($cmd);
$data = json_decode($result, true);

if (!$data || isset($data['error'])) {
    echo json_encode(["error" => "Prediction failed"]);
    exit;
}

$uid = $_SESSION['user_id'];

$conn->query("
INSERT INTO history(user_id, filename, prediction, confidence)
VALUES (
    '$uid',
    '$file',
    '{$data['result']}',
    '{$data['confidence']}'
)
");

echo json_encode($data);
