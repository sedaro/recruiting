# Debian-based node base image
FROM node:24-bookworm

# Add `/web/node_modules/.bin` to $PATH
ENV PATH /web/node_modules/.bin:$PATH

# Create and set working directory
RUN mkdir /web
WORKDIR /web

# Add web
COPY . .

# Install project dependencies
RUN npm install

# Start the Node app
CMD ["npm", "start"]
