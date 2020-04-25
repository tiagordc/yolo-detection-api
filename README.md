# yolo-detection-api

Object detection API with optional OCR based in a given trained model

Using Azure to store debug info

## Setup

```console
py -3.7 -m venv env
env\scripts\activate
py -m pip install --upgrade pip
pip install -r requirements.txt
```
## References

 * https://www.youtube.com/watch?v=_UqmgHKdntU
 * https://github.com/theAIGuysCode/Object-Detection-API
 
## TODO

 * Detection quadrands

## Docker test

Build:
 * docker build -t yolo-detection-api .

Run:
 * docker run -d -p 80:5000 --env-file ./.env yolo-detection-api
