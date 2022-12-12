# Use an official Python runtime as a parent image
FROM gcr.io/datamechanics/spark:platform-3.1-dm14

ENV PYSPARK MAJOR PYTHON VERSION=3

#create non-root user
#RUN sudo useradd -ms /bin/bash python-api
#USER python-api

# Set the working directory to /repo
WORKDIR /repo

# Copy the current directory contents into the container at /app
COPY . /repo

# Install any needed packages specified in requirements.txt
RUN python -m pip install -r requirements.txt
RUN pip install -y openjdk-8-jdk python3-venv bin/bash python-api

# define the command to run when the container is started
EXPOSE 8080
ENTRYPOINT [ "python" ]
CMD ["main.py]
