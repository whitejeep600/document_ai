# Setup

Required tools: conda, poetry, Docker.

To create the `document_ai` Python environment and install all the required packages, run

`conda env create -f environment.yaml`

`conda activate document_ai`

`poetry install --no-root`

Note: the local environment is needed only to run the demo described below. If the goal is only to run the app in the container, then it is not necessary because the app's environment is separate and set up inside the container.

# Running the demo

Build the Docker image with `make build` and run the container with `make run`. Wait for the container to start running the server (it should show `Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`). In a separate console, in the repository root, run

`python -m demo.send_demo_requests`

That script reads data from `demo/publaynet_example_data`, sends it to the endpoints, and saves the resulting metrics and plots under `demo/results`.

As the name suggests, the images, their annotations and the detected object categories used in this repo were taken from the PubLayNet dataset. The following color code was used for the categories:

- text: red,
- title: blue,
- list: green,
- table: purple,
- figure: yellow.

The detections from the model are plotted as rectangle edges, while the ground truth annotations are plotted as semi-transparent, filled rectangles. In addition to the images, metrics are saved in `.json` files.

After running the demo, `make stop` can be used to remove the container.

# Metrics

The detections from the used model and the annotations in the dataset are not entirely in the same format. For example, a single text detection from the model may cover a few contiguous paragraphs, while the annotations are separate bounding boxes for each paragraph, which cover the detection bounding box tightly. Thus the detections and annotations are not mapped one-to-one.

To account for this, pixel-wise IoU between the detections and labels, reported separately for each object category, was chosen as a simple and intuitive metric. The metric is undefined ("null" in the metric .jsons) if neither annotations nor detections for a given label are present in the image.

The task suggests scalar metrics only. In practice, it would also be informative to adjust the response format of the `evaluate` endpoint and create confusion matrices to illustrate how often the categories get confused for each other (e.g. title for simple text).

# Applied tools

The `PPStructure` tool from the open-source `paddleocr` package was used for the detection backbone.

Note that `paddlepaddle` requires a whole separate version of the package, `paddlepaddle-gpu` (linux-only), to support GPU, and it cannot fall back on CPU if it is not available. Since I had no private access to an appropriate environment during development, I have not been able to test the GPU version.

`fastapi` was used to create the HTTP endpoints.

Development was helped with linters: `mypy` for static type control, `black` and `isort` for formatting, and `flake8` for code style and quality.



