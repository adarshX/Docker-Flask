FROM continuumio/anaconda3:4.4.0
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000 
CMD python app_UI.py
