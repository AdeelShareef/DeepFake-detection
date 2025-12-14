<?php
session_start();
if (!isset($_SESSION['username'])) {
    header("Location: index.php");
    exit();
}
include('db_connection.php');

// Fetch History
$uid = $_SESSION['user_id'];
$sql = "SELECT * FROM history WHERE user_id='$uid' ORDER BY upload_time DESC LIMIT 5";
$history = $conn->query($sql);
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepShield Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="css/style.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <div class="col-md-2 sidebar">
                <h3 class="text-center neon-text mb-5">DEEPSHIELD</h3>
                <a href="#">Dashboard</a>
                <a href="#">Analysis</a>
                <a href="#">History</a>
                <a href="logout.php">Logout</a>
            </div>

            <div class="col-md-10 p-4">
                <h2 class="neon-text mb-4">Welcome, <?php echo $_SESSION['username']; ?></h2>

                <?php if(isset($_GET['result'])): ?>
                <div class="alert alert-info">
                    <strong>Analysis Complete:</strong> The media is classified as 
                    <span class="badge bg-<?php echo ($_GET['result'] == 'REAL' ? 'success' : 'danger'); ?> fs-5">
                        <?php echo $_GET['result']; ?>
                    </span>
                    with confidence: <?php echo $_GET['conf']; ?>
                </div>
                <?php endif; ?>

                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card p-3 text-center">
                            <h5>Model Accuracy</h5>
                            <h2 class="text-info">96.4%</h2>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card p-3 text-center">
                            <h5>F1 Score</h5>
                            <h2 class="text-warning">0.92</h2>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card p-3">
                            <h5>Upload Media for Scanning</h5>
                            <form action="upload_handler.php" method="post" enctype="multipart/form-data">
                                <div class="input-group">
                                    <input type="file" name="mediaFile" class="form-control" required>
                                    <button class="btn btn-primary" type="submit">SCAN NOW</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>

                <div class="row">
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-header">Performance Metrics (ROC Curve)</div>
                            <div class="card-body">
                                <canvas id="rocChart" height="100"></canvas>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header">Recent Scans</div>
                            <div class="card-body">
                                <table class="table table-dark table-sm">
                                    <thead><tr><th>File</th><th>Result</th></tr></thead>
                                    <tbody>
                                        <?php while($row = $history->fetch_assoc()): ?>
                                            <tr>
                                                <td><?php echo substr($row['filename'], 0, 15); ?>...</td>
                                                <td><?php echo $row['prediction']; ?></td>
                                            </tr>
                                        <?php endwhile; ?>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('rocChart').getContext('2d');
        const myChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                datasets: [{
                    label: 'ROC Curve',
                    data: [0, 0.8, 0.9, 0.95, 0.98, 1.0],
                    borderColor: '#e94560',
                    backgroundColor: 'rgba(233, 69, 96, 0.2)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                scales: {
                    x: { grid: { color: '#2a2a40' } },
                    y: { grid: { color: '#2a2a40' } }
                }
            }
        });
    </script>
</body>
</html>