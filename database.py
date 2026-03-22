import sqlite3

def create_db():
    conn=sqlite3.connect('users.db')
    cursor=conn.cursor()

    cursor.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,email TEXT NOT NULL,password TEXT NOT NULL)")
    
    conn.commit()
    conn.close()
    print("databae and users table created successfully")

if __name__=="__main__":
    create_db()