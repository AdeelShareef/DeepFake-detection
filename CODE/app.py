import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import sqlite3
from functools import wraps
from prediction import predict_image, predict_video

DB = 'deepshield.db'
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMG = {'png','jpg','jpeg'}
ALLOWED_VID = {'mp4','mov','avi'}

app = Flask(__name__)
app.secret_key = 'change_this_secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- DB helpers ----
def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

# simple login required decorator
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db()
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?,?)', (username, password))
            conn.commit()
        except Exception as e:
            return f"Error: {e}"
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db()
        cur = conn.execute('SELECT id FROM users WHERE username=? AND password=?', (username, password))
        row = cur.fetchone()
        if row:
            session['user_id'] = row['id']
            session['username'] = username
            return redirect(url_for('dashboard'))
        return 'Invalid credentials', 401
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

# upload and analyze
def allowed_file(filename, allowed):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in allowed

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    f = request.files.get('file')
    if not f:
        return 'No file', 400
    fname = f.filename
    if allowed_file(fname, ALLOWED_IMG.union(ALLOWED_VID)):
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        f.save(save_path)
        if allowed_file(fname, ALLOWED_IMG):
            pred, conf = predict_image(save_path)
            store_prediction(session['user_id'], fname, pred, conf)
            return jsonify({'filename': fname, 'pred': pred, 'confidence': conf})
        else:
            results = predict_video(save_path)
            # store aggregated result (e.g., majority over frames)
            avg_conf = sum([r['confidence'] for r in results]) / max(1, len(results))
            votes = sum([r['pred'] for r in results])
            pred = 1 if votes > (len(results)/2) else 0
            store_prediction(session['user_id'], fname, pred, avg_conf)
            return jsonify({'filename': fname, 'pred': pred, 'confidence': avg_conf, 'frames': results})
    else:
        return 'Not allowed file type', 400

def store_prediction(user_id, filename, predicted_label, confidence):
    conn = get_db()
    conn.execute('INSERT INTO predictions (user_id, filename, predicted_label, confidence) VALUES (?,?,?,?)',
                 (user_id, filename, predicted_label, confidence))
    conn.commit()

@app.route('/metrics')
@login_required
def metrics():
    # Calculates accuracy, f1 and ROC data using stored predictions with ground_truth not null
    conn = get_db()
    cur = conn.execute('SELECT predicted_label, confidence, ground_truth FROM predictions WHERE ground_truth IS NOT NULL')
    rows = cur.fetchall()
    if not rows:
        return jsonify({'message': 'No labeled predictions yet', 'data': {}})
    y_true = [r['ground_truth'] for r in rows]
    y_pred = [r['predicted_label'] for r in rows]
    probs = [r['confidence'] for r in rows]
    from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_auc = float(auc(fpr, tpr))
    return jsonify({'accuracy': acc, 'f1': f1, 'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc}})

# admin helper to set ground truth for a prediction (for demo)
@app.route('/set_ground_truth', methods=['POST'])
@login_required
def set_ground_truth():
    pid = request.form['prediction_id']
    gt = int(request.form['ground_truth'])
    conn = get_db()
    conn.execute('UPDATE predictions SET ground_truth=? WHERE id=?', (gt, pid))
    conn.commit()
    return 'ok'

if __name__ == '__main__':
    app.run(debug=True)