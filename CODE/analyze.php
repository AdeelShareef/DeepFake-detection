<?php
// Simple forwarder that posts uploaded file to Flask backend (/upload)
// Place this if you have an existing PHP site and want to stay with PHP for frontend.
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['file'])) {
    $url = 'http://127.0.0.1:5000/upload';
    $cfile = curl_file_create($_FILES['file']['tmp_name'], $_FILES['file']['type'], $_FILES['file']['name']);
    $data = array('file' => $cfile);
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    $response = curl_exec($ch);
    curl_close($ch);
    header('Content-Type: application/json');
    echo $response;
}