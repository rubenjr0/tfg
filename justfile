init:
    uv sync

    # Get weights
    if [ -d "checkpoints" ]; then \
        echo "El directorio checkpoints ya existe, omitiendo descarga de pesos"; \
    else \
        mkdir -p checkpoints; \
        wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P checkpoints; \
    fi

    # Get data
    if [ -d "data" ]; then \
        echo "El directorio data ya existe, omitiendo descarga de datos"; \
    else \
        mkdir data; \
        cat data.txt | xargs -n 1 -P 4 wget -P data; \
        unzip "data/*.zip" -d data; \
        rm data/*.zip; \
    fi

    # Preprocess data
    uv run preprocess
