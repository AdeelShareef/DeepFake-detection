<?php
// db_connection.php
$servername = "localhost";
$username = "root";
$password = ""; // Default XAMPP password is empty
$dbname = "deepshield_db";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
?>