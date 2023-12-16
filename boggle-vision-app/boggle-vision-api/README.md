#  Boggle Vision API
This folder contains a FastAPI that can control the backend of the Boggle Vision app! 

---
### Running Locally

Below, I've included a command for running the Boggle Vision API locally: 

```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---
### Managing the Docker Image

The `Dockerfile` in this repo is a containerized version of the app. In order to run it locally, you can do the following: 

```
docker build -t boggle-vision-api .
docker run -p 8000:8000 boggle-vision-api
```

Here's how you can update the Docker image: 

```
docker tag boggle-vision-api us-central1-docker.pkg.dev/boggle-vision/boggle-vision-webapp/boggle-vision-api
docker push us-central1-docker.pkg.dev/boggle-vision/boggle-vision-webapp/boggle-vision-api
```

---

