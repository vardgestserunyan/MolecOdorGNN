FROM continuumio/miniconda3

LABEL maintainer="vardgestserunyan@gmail.com"

WORKDIR /app

COPY ./ ./

# Create environment from environment.yaml
RUN cat /app/environment.yaml
RUN conda env create -f environment_lean.yaml --name graph_molec_lean
EXPOSE 8080

CMD ["bash", "-c", "source activate graph_molec_lean && gunicorn --bind 0.0.0.0:$PORT gnn_model:app"]



