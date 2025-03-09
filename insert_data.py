import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get database credentials
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")

# Ensure all required variables are loaded
if not all([db_name, db_user, db_password, db_host, db_port]):
    raise ValueError("Missing database credentials. Check your .env file!")

# Database connection
conn = psycopg2.connect(
    dbname=db_name,
    user=db_user,
    password=db_password,
    host=db_host,
    port=db_port
)

cursor = conn.cursor()

# Insert sample data
cursor.execute("""
    INSERT INTO achievements ("user", message) 
    VALUES (%s, %s);
""", ("Bhavya", "Completed micro recognition setup successfully!"))

conn.commit()
print("Data inserted successfully!")

cursor.close()
conn.close()
