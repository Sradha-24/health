import sqlite3
conn=sqlite3.connect('users.db')
#conn.execute('''
#CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY AUTOINCREMENT,email TEXT,date TEXT, glucose REAL,bmi REAL,score REAL,result TEXT)
#''')
#conn.execute("CREATE TABLE IF NOT EXISTS hypertension_history(id INTEGER PRIMARY KEY AUTOINCREMENT,email TEXT,date TEXT,salt_intake REAL,score REAL,result TEXT)")
#conn.close()

conn.execute('''CREATE TABLE IF NOT EXISTS heart_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    date TEXT,
    cholesterol REAL,
    crp_level REAL,
    score REAL,
    result TEXT
)
''')
print("table created")