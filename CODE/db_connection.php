<?php
$conn = new mysqli("localhost", "root", "", "deepshield_db");
if ($conn->connect_error) {
    die("Database connection failed");
}
?>
