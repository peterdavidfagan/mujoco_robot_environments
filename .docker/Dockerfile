FROM python:3.10.6
RUN mkdir /app 
COPY . /app
WORKDIR /app
ENV PYTHONPATH=${PYTHONPATH}:${PWD} 
RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN git config --global url."https://github.com/".insteadOf git@github.com: && poetry install
WORKDIR /app/mujoco_robotics_environments/tasks
