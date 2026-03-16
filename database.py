import sqlite3

# Connect to a local file named "eco_users.db"
# If it doesn't exist, Python creates it automatically.
DB_NAME = "eco_users.db"

def init_db():
    """Create the table if it doesn't exist"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create a table to hold User Data
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            score INTEGER DEFAULT 0,
            scans_count INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username):
    """Register a new user (or ignore if they exist)"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # INSERT OR IGNORE means: if user exists, do nothing.
    c.execute('INSERT OR IGNORE INTO users (username, score, scans_count) VALUES (?, 0, 0)', (username,))
    conn.commit()
    conn.close()

def update_score(username, points):
    """Add points to a user"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('UPDATE users SET score = score + ?, scans_count = scans_count + 1 WHERE username = ?', (points, username))
    conn.commit()
    conn.close()

def get_user_stats(username):
    """Get current score and scan count"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT score, scans_count FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    return data if data else (0, 0) # Return 0 if new user

def get_leaderboard():
    """Get top 5 players"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT username, score FROM users ORDER BY score DESC LIMIT 5')
    data = c.fetchall()
    conn.close()
    return data