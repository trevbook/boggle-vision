#!/bin/bash

# Echo the `date` to the ~/date.txt file (this is a test)
echo $(date) >>~/date.txt

# Check if Docker is installed and install it if necessary
if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."

    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    # Add the repository to Apt sources:
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |
        sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    sudo apt-get update

    # Now, we'll install Docker
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

fi

# Now, we're going to authenticate Docker with GAR
echo "Configuring Docker for GAR authentication..."
export GOOGLE_APPLICATION_CREDENTIALS=$GCR_JSON_KEY
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
gcloud auth configure-docker us-central1-docker.pkg.dev

# Pull the latest Docker image
echo "Pulling the latest Docker image..."
docker pull us-central1-docker.pkg.dev/boggle-vision/boggle-vision-webapp/boggle-vision-ui:latest

# Stop and remove the current running container
echo "Stopping and removing current container..."
docker stop boggle-vision-ui-container || true
docker rm boggle-vision-ui-container || true

# Run the new container
echo "Running the new container..."
docker run --name boggle-vision-ui-container -d us-central1-docker.pkg.dev/boggle-vision/boggle-vision-webapp/boggle-vision-ui:latest
