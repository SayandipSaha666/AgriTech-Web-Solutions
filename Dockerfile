# 1️⃣ Use an official Python image
FROM python:3.10

# 2️⃣ Set the working directory in the container
WORKDIR /app

# 3️⃣ Disable GPU usage (if not needed)
ENV CUDA_VISIBLE_DEVICES=-1

# 4️⃣ Install TensorFlow 2.19.0 to match model training version
RUN pip install tensorflow==2.19.0

# 5️⃣ Copy everything from your project folder to /app
COPY . .

# 6️⃣ Install other dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 7️⃣ Expose the port Flask runs on
EXPOSE 8080

# 8️⃣ Start the application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
