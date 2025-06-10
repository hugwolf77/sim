
from pydantic import BaseModel  

# create dataclass 
class BookCreate(BaseModel):  
	title: str  
	author: str  
  
class Book(BookCreate):  
	id: int