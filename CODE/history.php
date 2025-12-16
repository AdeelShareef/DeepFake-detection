<?php
session_start();
include('db_connection.php');
$uid = $_SESSION['user_id'];
$res = $conn->query("SELECT * FROM history WHERE user_id='$uid' ORDER BY upload_time DESC");
?>
<!DOCTYPE html>
<html>
<head>
<title>History</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="css/style.css" rel="stylesheet">
</head>
<body>
<div class="container p-4">
<h3 class="neon-text">Scan History</h3>
<table class="table table-dark">
<tr><th>File</th><th>Result</th><th>Confidence</th><th>Date</th></tr>
<?php while($r=$res->fetch_assoc()): ?>
<tr>
<td><?php echo $r['filename']; ?></td>
<td><?php echo $r['prediction']; ?></td>
<td><?php echo round($r['confidence']*100,2); ?>%</td>
<td><?php echo $r['upload_time']; ?></td>
</tr>
<?php endwhile; ?>
</table>
<a href="dashboard.php" class="btn btn-primary">Back</a>
</div>
</body>
</html>
