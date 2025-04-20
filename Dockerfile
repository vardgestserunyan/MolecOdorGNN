FROM continuumio/miniconda3

LABEL maintainer="vardgestserunyan@gmail.com"

WORKDIR /app

COPY ./ ./

# Create environment from environment.yaml
RUN cat /app/environment.yaml
RUN conda env create -f environment_lean.yaml --name graph_molec_lean

CMD ["bash", "-c", "source activate graph_molec_lean && exec python3 gnn_model.py"]



