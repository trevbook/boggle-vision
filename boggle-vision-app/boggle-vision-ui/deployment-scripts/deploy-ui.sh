#!/bin/bash

# Echo the `date` to the ~/date.txt file (this is a test)
echo $(date) >> ~/date.txt

# Check if Docker is installed and install it if necessary
# if ! command -v docker &> /dev/null
# then
#     echo "Installing Docker..."
#     sudo apt-get update
#     sudo apt-get install docker-ce docker-ce-cli containerd.io
# fi

# # Pull the latest Docker image
# echo "Pulling the latest Docker image..."
# docker pull us-central1-docker.pkg.dev/boggle-vision/boggle-vision-webapp/boggle-vision-ui:latest

# # Stop and remove the current running container
# echo "Stopping and removing current container..."
# docker stop boggle-vision-ui-container || true
# docker rm boggle-vision-ui-container || true

# # Run the new container
# echo "Running the new container..."
# docker run --name boggle-vision-ui-container -d us-central1-docker.pkg.dev/boggle-vision/boggle-vision-webapp/boggle-vision-ui:latest
