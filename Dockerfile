# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . .

# Step 4: Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Set environment variables (optional, if using .env file)
# You can add more if needed (e.g., database credentials)
ENV MODEL_PATH=/app/models/bert_intent_classifier

# Step 6: Expose port 5000 for the Flask app
EXPOSE 5000

# Step 7: Run Gunicorn to serve the Flask app
CMD ["gunicorn", "-w", "4", "app:app"]
