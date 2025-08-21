from fastapi import FastAPI

app = FastAPI(title='Deepfake Detection API')

@app.get('/')
def health():
    return {'status': 'ok'}
