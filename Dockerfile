# Use the lightweight Micromamba image
FROM mambaorg/micromamba:1.4.0

# 1) Copy your environment spec
WORKDIR /app
COPY environment.yml .

# 2) Create the Conda env via Micromamba
RUN micromamba create -y -n aes_project -f environment.yml \
    && micromamba clean --all --yes

# 3) Activate it by default
ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    CONDA_DEFAULT_ENV=aes_project \
    PATH=/opt/conda/envs/aes_project/bin:$PATH \
    PORT=8501 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# 4) Pre-download the spaCy model
RUN micromamba run -n aes_project python -m spacy download en_core_web_sm

# 5) Bake in your model artifacts and code
COPY models/ /app/models/
COPY src/ /app/src/

EXPOSE 8501
ENTRYPOINT ["bash","-lc","micromamba run -n aes_project streamlit run src/app.py"]