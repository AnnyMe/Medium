FROM continuumio/anaconda3
COPY ./copied_to_docker ./work_dir
EXPOSE 5000
WORKDIR ./work_dir
RUN pip install -r requirements.txt
CMD python flask_api.py
