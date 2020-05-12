# yolo-detection-api

Object detection API with OCR (optional) based in a given trained model

Also using Azure optionally to:
 * Store the tensorflow model
 * Store images and predictions
 * API key authorization

## Setup (Windows)

```console
py -3.7 -m venv env
env\scripts\activate
py -m pip install --upgrade pip
pip install -r requirements.txt
```
## References

 * https://www.youtube.com/watch?v=_UqmgHKdntU
 * https://github.com/theAIGuysCode/Object-Detection-API
 
## Docker cheat sheet

Build:
 * docker build -t yolo-detection-api .

Run:
 * docker run -d -p 80:5000 --env-file ./.env yolo-detection-api

Push:
 * https://ropenscilabs.github.io/r-docker-tutorial/04-Dockerhub.html
 
