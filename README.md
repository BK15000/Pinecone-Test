Download dataset here(books_rating.csv):

https://huggingface.co/datasets/rootstrap-org/books-ratings/viewer

TO RUN:

Setup .env using example.env


Make venv:
python3 -m venv venv


Activate:
	Mac:
	source venv/bin/activate
	  
	Windows:
	venv\Scripts\activate

Install pinecone:
pip3 install pinecone
pip3 install pandas
pip3 install dotenv
pip3 install os

Run code to load data:
python3 pinecone_data.py

Run code to query data:
python3 pinecone_app.py








