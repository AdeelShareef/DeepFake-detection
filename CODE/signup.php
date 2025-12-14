<?php include('db_connection.php'); ?>
<!DOCTYPE html>
<html>
<head>
    <title>DeepShield - Signup</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="css/style.css" rel="stylesheet">
</head>
<body class="d-flex align-items-center justify-content-center" style="height: 100vh;">
    <div class="card p-4" style="width: 400px;">
        <h3 class="text-center neon-text mb-4">DEEPSHIELD</h3>
        <?php
        if ($_SERVER["REQUEST_METHOD"] == "POST") {
            $user = $_POST['username'];
            $pass = password_hash($_POST['password'], PASSWORD_DEFAULT);
            $sql = "INSERT INTO users (username, password) VALUES ('$user', '$pass')";
            if ($conn->query($sql) === TRUE) {
                echo "<div class='alert alert-success'>Account created! <a href='index.php'>Login</a></div>";
            } else {
                echo "<div class='alert alert-danger'>Error: " . $conn->error . "</div>";
            }
        }
        ?>
        <form method="post">
            <div class="mb-3">
                <label>Username</label>
                <input type="text" name="username" class="form-control" required>
            </div>
            <div class="mb-3">
                <label>Password</label>
                <input type="password" name="password" class="form-control" required>
            </div>
            <button type="submit" class="btn btn-primary w-100">Sign Up</button>
            <div class="mt-3 text-center">
                <a href="index.php" class="text-muted">Already have an account? Login</a>
            </div>
        </form>
    </div>
</body>
</html>