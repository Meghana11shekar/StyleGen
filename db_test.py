from supabase import create_client
import os
from dotenv import load_dotenv

# Load the .env file (which stores your keys safely)
load_dotenv()

# Get Supabase credentials from .env
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# Create a Supabase client (connect to your project)
supabase = create_client(url, key)

# Try reading from your 'items' table
data = supabase.table("items").select("*").execute()

print("✅ Connection successful!")
print("Data from items table:", data.data)
