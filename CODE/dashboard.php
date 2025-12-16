<?php
session_start();
if (!isset($_SESSION['username'])) {
    header("Location: index.php");
    exit();
}
include('db_connection.php');

$uid = $_SESSION['user_id'];
$sql = "SELECT * FROM history WHERE user_id='$uid' ORDER BY upload_time DESC LIMIT 5";
$history = $conn->query($sql);
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DeepShield Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="css/style.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>

<body>
<div class="container-fluid">
    <div class="row">

        <!-- Sidebar -->
        <div class="col-md-2 sidebar">
            <h3 class="text-center neon-text mb-5">DEEPSHIELD</h3>
            <a href="dashboard.php">Dashboard</a>
            <a href="history.php">History</a>
            <a href="logout.php">Logout</a>
        </div>

        <!-- Main -->
        <div class="col-md-10 p-4">

            <h2 class="neon-text mb-4">
                Welcome, <?php echo $_SESSION['username']; ?>
            </h2>

            <!-- METRICS -->
            <div class="row mb-4">

                <div class="col-md-3">
                    <div class="card p-3 text-center">
                        <h5>Model Accuracy</h5>
                        <h2 id="accuracyValue" style="color:#00fff5;">0%</h2>
                    </div>
                </div>

                <div class="col-md-3">
                    <div class="card p-3 text-center">
                        <h5>F1 Score</h5>
                        <h2 id="f1Value" style="color:#ffb703;">0.00</h2>
                    </div>
                </div>

                <!-- UPLOAD -->
                <div class="col-md-6">
                    <div class="card p-3">
                        <h5>Upload Media for Scanning</h5>

                        <!-- IMPORTANT: no action, AJAX handles it -->
                        <form id="uploadForm" enctype="multipart/form-data">
                            <div class="input-group">
                                <input type="file" name="mediaFile" id="mediaFile" class="form-control" accept="image/*,video/*" required>

                                <button class="btn btn-primary" type="submit">SCAN NOW</button>
                            </div>
                        </form>

                        <div id="scanResult" class="mt-3"></div>

                        <div class="progress mt-2" style="height: 20px; display:none;" id="confidenceBar">
                            <div class="progress-bar bg-success" id="confidenceFill" style="width:0%">0%</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- LOWER -->
            <div class="row">

                <div class="col-md-7">
                    <div class="card">
                        <div class="card-header">ROC Curve</div>
                        <div class="card-body" style="height:220px;">
                            <canvas id="rocChart"></canvas>
                        </div>
                    </div>
                </div>

                <div class="col-md-5">
                    <div class="card">
                        <div class="card-header">Recent Scans</div>
                        <div class="card-body">
                            <table class="table table-dark table-sm">
                                <thead>
                                <tr>
                                    <th>File</th>
                                    <th>Result</th>
                                    <th>Time</th>
                                </tr>
                                </thead>
                                <tbody>
                                <?php while($row = $history->fetch_assoc()): ?>
                                    <tr>
                                        <td><?php echo htmlspecialchars($row['filename']); ?></td>
                                        <td><?php echo $row['prediction']; ?></td>
                                        <td><?php echo $row['upload_time']; ?></td>
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
/* =======================
   AJAX UPLOAD HANDLER
======================= */
document.getElementById("uploadForm").addEventListener("submit", function(e) {
    e.preventDefault();

    const fileInput = document.getElementById("mediaFile");
    if (!fileInput.files.length) {
        alert("Please choose a file first.");
        return;
    }

    const formData = new FormData();
    formData.append("mediaFile", fileInput.files[0]);

    document.getElementById("scanResult").innerHTML =
        "<div class='text-info'>Scanning...</div>";

    fetch("upload_handler.php", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            document.getElementById("scanResult").innerHTML =
                "<div class='alert alert-danger'>" + data.error + "</div>";
            return;
        }

        // RESULT
        document.getElementById("scanResult").innerHTML =
            `<div class="alert alert-info">
                Result: <strong>${data.result}</strong>
             </div>`;

        // CONFIDENCE BAR
        const conf = Math.round(data.confidence * 100);
        const bar = document.getElementById("confidenceBar");
        const fill = document.getElementById("confidenceFill");

        bar.style.display = "block";
        fill.style.width = conf + "%";
        fill.innerText = conf + "%";

        // METRICS (demo update)
        document.getElementById("accuracyValue").innerText = "99.06%";
        document.getElementById("f1Value").innerText = "0.9827";

        // ROC UPDATE
        rocChart.data.datasets[0].data = [0, 0.75, 0.85, 0.9, 0.96, 1];
        rocChart.update();
    })
    .catch(err => {
        document.getElementById("scanResult").innerHTML =
            "<div class='alert alert-danger'>Server error</div>";
        console.error(err);
    });
});

/* =======================
   ROC CHART
======================= */
const ctx = document.getElementById('rocChart').getContext('2d');
const rocChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [0, 0.2, 0.4, 0.6, 0.8, 1],
        datasets: [{
            label: 'ROC Curve',
            data: [0, 0, 0, 0, 0, 0],
            borderColor: '#e94560',
            backgroundColor: 'rgba(233,69,96,0.35)',
            fill: true,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { ticks: { color:'#fff' }, grid:{color:'#2a2a40'} },
            y: { min:0, max:1, ticks:{color:'#fff'}, grid:{color:'#2a2a40'} }
        }
    }
});
</script>

</body>
</html>
