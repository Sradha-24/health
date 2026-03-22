import sqlite3
conn=sqlite3.connect('users.db')
conn.execute('INSERT INTO users(name,email,password) VALUES("admin","admin@gmail.com","admin")')
conn.commit()
conn.close()

print("admin created")