create docker image with image name ml-api:
- docker build -t ml-api .

run docker: 
- docker run -p 5000:5000 ml-api