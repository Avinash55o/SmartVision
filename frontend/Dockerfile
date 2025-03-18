# Stage 1: Build
FROM node:18-alpine AS builder
WORKDIR /app

# Copy package.json and package-lock.json first
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the entire code
COPY . .

# Build the Next.js app
RUN npm run build

# Stage 2: Serve
FROM node:18-alpine
WORKDIR /app

# Copy the built files from the previous stage
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package*.json ./

# Install only production dependencies
RUN npm install --only=production

# Start the Next.js app
CMD ["npm", "run", "start"]
