# Mpi and spark for cosine similarity (via tf-idf vectorization)

Project to check processing time using mpi and spark for text mining.


## How to run mpi?
1. Clone the repo.
2. Create virtualenv with Python 3.9
3. Activate virtualenv.
4. Install dependencies.
5. Call make run

```console
https://github.com/RafaDias/mpi-and-spark-for-tf-idf.git mpi_and_spark
cd mpi_and_spark
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
make run 
```

## How to run spark in jupyter notebook?
1. Clone the repo.
2. Create jupyter notebook docker image.

```console
https://github.com/RafaDias/mpi-and-spark-for-tf-idf.git mpi_and_spark
cd mpi_and_spark
docker run -it -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter/pyspark-notebook
```