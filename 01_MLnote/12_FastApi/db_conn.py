
import sqlite3
from dataclass import BookCreate, Book

# create sqlite3 DB connect def   
def create_connection():  
	connection = sqlite3.connect("./books.db")  
	return connection

def create_table():  
	connection = create_connection()  
	cursor = connection.cursor()  
	cursor.execute("""  
			CREATE TABLE IF NOT EXISTS books (  
			id INTEGER PRIMARY KEY AUTOINCREMENT,  
			title TEXT NOT NULL,  
			author TEXT NOT NULL  
			)  
		""")  
	connection.commit()  
	connection.close()  
	  
# create_table() # Call this function to create the table


# Insert data into DB use dataclass 
def create_book(book: BookCreate):  
	connection = create_connection()  
	cursor = connection.cursor()  
	cursor.execute(
				"INSERT INTO books (title, author) VALUES (?, ?)",
				(book.title, book.author)
				)  
	connection.commit()  
	connection.close()
	
def read_books(book_id: int):
	connection = create_connection()
	cursor = connection.cursor()
	cursor.execute("SELECT title, author FROM books WHERE id = ?", (book_id,))
	book = cursor.fetchone()
	connection.close()
	return book