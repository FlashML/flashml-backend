APP=flash-backend

docker-build: 
	docker build -t ${APP} .

docker-run:
	docker run -d -p 5000:5000 ${APP}

docker-build-and-run: docker-build docker-run
