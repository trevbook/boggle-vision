# Boggle Vision UI
This folder contains all of the code necessary for the Boggle Vision UI. 

---
### Managing the Docker Image
In order to run the UI's Docker image locally, you can do the following: 

```
docker build -t boggle-vision-ui .
docker run -p 3000:3000 boggle-vision-ui
```

Here's how you can update the Docker image: 

```
docker tag boggle-vision-ui us-central1-docker.pkg.dev/boggle-vision/boggle-vision-webapp/boggle-vision-ui
docker push us-central1-docker.pkg.dev/boggle-vision/boggle-vision-webapp/boggle-vision-ui
```

---