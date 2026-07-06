from fastapi import FastAPI
from src.api.endpoints import api_router
from src.api.middleware import RequestIDMiddleware

app = FastAPI(title="AI Engineering API")

app.add_middleware(RequestIDMiddleware)
app.include_router(api_router)


@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
