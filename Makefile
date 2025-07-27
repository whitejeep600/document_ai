IMAGE_NAME = document_ai_image
CONTAINER_NAME = document_ai_container
PORT = 8000

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -d -p $(PORT):$(PORT) --name $(CONTAINER_NAME)  $(IMAGE_NAME)

stop:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

rebuild: stop build run

