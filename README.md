# Story Point Estimator System

This application takes input of title and description of an issue and estimates the number of story points required to complete it.

## How to use

1. Resolve the environment

```
pip install -r requirements.txt

```

2. Start Streamlit server
```
streamlit run app.py
```

### Deploy with Docker
Manually deployment
```
docker build -t spes -f Dockerfile.spes .
docker run --name spes -p 8501:8501 -d spes
```

Or you can use *docker-compose* by setting the *PORT* in [.env](.env) file and then run
```
docker-compose up -d
```