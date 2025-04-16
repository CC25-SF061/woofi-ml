FROM --platform=linux/x86_64 python:3-slim
 
 WORKDIR /usr/src/app
 COPY requirements.txt ./
 RUN apt-get update && apt-get install -y libpq-dev build-essential
 RUN pip install fastapi uvicorn scikit-learn pandas numpy googletrans==4.0.0-rc1 joblib legacy-cgi
 COPY . .
 CMD ["uvicorn","--host","0.0.0.0","main:app"]