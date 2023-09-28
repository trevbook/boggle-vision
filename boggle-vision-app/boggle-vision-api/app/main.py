# This is the main entrypoint to the Boggle Vision API.

# ============================================
#                    SETUP
# ============================================
# The code below will set up the rest of this file.

# Import statements
from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Setting up the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
#                 ENDPOINTS
# ============================================
# Below, we'll define all of the endpoints for the API.


@app.get("/")
def read_root():
    print("Hey.")
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


# The following endpoint is a POST endpoint that will
# take in data and then return the JSON representation
# of that data.
@app.post("/analyze_image")
def read_data(data: dict):
    return {"this is a key": "this is a value"}
