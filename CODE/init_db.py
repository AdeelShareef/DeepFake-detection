import sqlite3


conn = sqlite3.connect('deepshield.db')
cur = conn.cursor()


cur.executescript('''
CREATE TABLE IF NOT EXISTS users (
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT UNIQUE NOT NULL,
password TEXT NOT NULL -- WARNING: plain text for demo. Use hashing in production!
);


CREATE TABLE IF NOT EXISTS predictions (
id INTEGER PRIMARY KEY AUTOINCREMENT,
user_id INTEGER,
filename TEXT,
predicted_label INTEGER, -- 0 = real, 1 = manipulated
confidence REAL,
ground_truth INTEGER, -- optional: 0/1 if user supplies label
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
FOREIGN KEY(user_id) REFERENCES users(id)
);
''')


conn.commit()
conn.close()
print('Initialized database deepshield.db')