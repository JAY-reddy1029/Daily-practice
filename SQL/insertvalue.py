import mysql.connector

conn=mysql.connector.connect(host="localhost",user="root",password="9880",database="pythondb")

mycursor=conn.cursor()

sql='insert into student (name,branch,id) values (%s,%s,%s)'
#val=('jhon','cse',56)

val=[('jhon','cse',56),('doe','ece',34),('smith','mech',23)]

mycursor.executemany(sql,val)
conn.commit()
print(mycursor.rowcount,"record inserted")