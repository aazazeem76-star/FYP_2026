import sqlite3
import pickle
import os

db_path = r'e:\FRA\db.sqlite3'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Find teacher3
cur.execute("SELECT id, username, role, first_name, last_name FROM attendance_app_user WHERE username='teacher3'")
row = cur.fetchone()
if row:
    print(f"teacher3: id={row[0]}, role={row[2]}, name={row[3]} {row[4]}")
    teacher_id = row[0]
else:
    print("teacher3 NOT FOUND")
    teacher_id = None

# Count biometric samples for teacher3
if teacher_id:
    cur.execute("SELECT COUNT(*) FROM attendance_app_biometricsample WHERE user_id=? AND sample_type='face'", (teacher_id,))
    count = cur.fetchone()[0]
    print(f"teacher3 biometric face samples: {count}")

# List all users and their face sample counts
print("\nAll users with face samples:")
cur.execute("""
    SELECT u.id, u.username, u.role, COUNT(b.id) as samples
    FROM attendance_app_user u
    LEFT JOIN attendance_app_biometricsample b ON b.user_id = u.id AND b.sample_type='face'
    GROUP BY u.id ORDER BY u.id
""")
for r in cur.fetchall():
    print(f"  id={r[0]}, username={r[1]}, role={r[2]}, face_samples={r[3]}")

conn.close()

# Read label encoders from pkl files
print("\nLabel encoders in model files:")
models_dir = r'e:\FRA\media\models'
for fname in sorted(os.listdir(models_dir)):
    if fname.endswith('_labels.pkl'):
        fpath = os.path.join(models_dir, fname)
        with open(fpath, 'rb') as f:
            labels = pickle.load(f)
        print(f"  {fname}: {labels}")
