# I'm using a multi-stage build here, as suggested by 
# my robot coworker, Chat-GPT. Here's the conversation: 
# https://chat.openai.com/share/001d6284-8be9-47e9-a62c-d1585116e49c

# Stage 1: Build
FROM node:18.13.0-alpine as builder

WORKDIR /usr/src/app

COPY package*.json ./

RUN npm install

COPY . .

RUN npm run build

# Stage 2: Run
FROM node:18.13.0-alpine

WORKDIR /usr/src/app

# Install 'serve' in the production image
RUN npm install -g serve

# Copy only necessary files from the builder stage
COPY --from=builder /usr/src/app/build ./build

EXPOSE 3000

CMD ["serve", "-s", "build", "-l", "3000"]
