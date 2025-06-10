from fastapi import FastAPI  
import db_conn  
from dataclass import BookCreate, Book
 
# create_table() # Call this function to create the table
# db_conn.create_table()

# create API
app = FastAPI()  
  
@app.get("/")  
def read_root():  
	return {"message": "Welcome to the CRUD API"}

@app.get("/read_books/{book_id}")
def read_book(book_id: int):
	getbook = db_conn.read_books(book_id)
	print(getbook)
	return {"book_id" : book_id, "title" : getbook[0], "author" : getbook[1]}

@app.post("/books/")  
def create_book(book: BookCreate):  
	db_conn.create_book(book)  
	return book
