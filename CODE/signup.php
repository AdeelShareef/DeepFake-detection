<?php
include('db_connection.php');

if ($_SERVER["REQUEST_METHOD"] === "POST") {
    $u = $_POST['username'];
    $p = password_hash($_POST['password'], PASSWORD_DEFAULT);

    if ($conn->query("INSERT INTO users(username,password) VALUES('$u','$p')")) {
        $success = true;
    } else {
        $error = "Username already exists";
    }
}
?>
<!DOCTYPE html>
<html>
<head>
<title>Signup</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="css/style.css" rel="stylesheet">
</head>
<body class="d-flex justify-content-center align-items-center vh-100">
<div class="card p-4" style="width:400px;">
<h3 class="text-center neon-text mb-3">SIGN UP</h3>
<?php if(isset($success)): ?>
<div class="alert alert-success">
Account created successfully!
<a href="index.php" class="btn btn-sm btn-light mt-2">Back to Login</a>
</div>
<?php else: ?>
<?php if(isset($error)) echo "<div class='alert alert-danger'>$error</div>"; ?>
<form method="post">
<input name="username" class="form-control mb-3" placeholder="Username" required>
<input name="password" type="password" class="form-control mb-3" placeholder="Password" required>
<button class="btn btn-primary w-100">Create Account</button>
</form>
<a href="index.php" class="btn btn-outline-light w-100 mt-2">Back to Login</a>
<?php endif; ?>
</div>
</body>
</html>
