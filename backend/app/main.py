from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="Football Prediction API",
    description="API for predicting football match outcomes",
    version="1.0.0"
)

# Set up CORS - include both localhost:3000 and any additional frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # Default React port
        "http://localhost:3001",    # Alternative React port
        "http://localhost:3002",    # Another possible React port
        "http://localhost:3003",    # Another possible React port
        "http://localhost:3004",    # Another possible React port
        "http://127.0.0.1:3000",    # Using IP instead of localhost
        "http://192.168.1.100:3000" # Possible local network IP
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import API endpoints
from app.api.endpoints import router as api_router

# Include API router - keep your original prefix
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Football Prediction API is running. Access /docs for documentation."}

# Run the application with uvicorn if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)