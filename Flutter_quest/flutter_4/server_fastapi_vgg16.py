# server_fastapi_vgg16.py
import uvicorn   # pip install uvicorn 
from fastapi import FastAPI   # pip install fastapi 
from fastapi.middleware.cors import CORSMiddleware # 추가된부분 cors 문제 해결을 위한

# 예측 모듈 가져오기
import vgg16_prediction_model

# Create the FastAPI application
app = FastAPI()

# cors 이슈
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A simple example of a GET request
@app.get("/")
async def read_root():
    print("url was requested")
    return "VGG16모델을 사용하는 API를 만들업 봅시다."

@app.get('/sample1')
async def sample_prediction():
    result = await vgg16_prediction_model.prediction_model()
    print(result["predicted_label"])
    return result

@app.get('/sample2')
async def sample_prediction():
    result = await vgg16_prediction_model.prediction_model()
    print(result["prediction_score"])
    return result


# Run the server
if __name__ == "__main__":
    uvicorn.run("server_fastapi_vgg16:app",
            reload= True,   # Reload the server when code changes
            host="127.0.0.1",   # Listen on localhost 
            port=5000,   # Listen on port 5000 
            log_level="info"   # Log level
            )