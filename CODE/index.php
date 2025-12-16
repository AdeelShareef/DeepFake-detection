<?php
session_start();
include('db_connection.php');

if ($_SERVER["REQUEST_METHOD"] === "POST") {
    $u = $_POST['username'];
    $p = $_POST['password'];

    $res = $conn->query("SELECT * FROM users WHERE username='$u'");
    if ($res->num_rows === 1) {
        $row = $res->fetch_assoc();
        if (password_verify($p, $row['password'])) {
            $_SESSION['username'] = $u;
            $_SESSION['user_id'] = $row['id'];
            header("Location: dashboard.php");
            exit();
        }
        $error = "Invalid password";
    } else {
        $error = "User not found";
    }
}
?>
<!DOCTYPE html>
<html>
<head>
<title>DeepShield Login</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="css/style.css" rel="stylesheet">
</head>
<body class="d-flex justify-content-center align-items-center vh-100">
<div class="card p-4" style="width:400px;">
<h3 class="text-center neon-text mb-3">DEEPSHIELD LOGIN</h3>
<?php if(isset($error)) echo "<div class='alert alert-danger'>$error</div>"; ?>
<form method="post">
<input name="username" class="form-control mb-3" placeholder="Username" required>
<input name="password" type="password" class="form-control mb-3" placeholder="Password" required>
<button class="btn btn-primary w-100">Login</button>
<a href="signup.php" class="btn btn-outline-light w-100 mt-2">Sign Up</a>
</form>
</div>
</body>
</html>
