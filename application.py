import time, cv2, numpy as np, tensorflow as tf, os, random, string, pytesseract, re, urllib, zipfile, io
from flask import Flask, request, jsonify, abort, send_file, send_from_directory
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.cosmosdb.table.tableservice import TableService
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.models import YoloV3

IMAGE_SIZE = int(os.environ['IMAGE_SIZE'])
CLASS_NAMES = os.environ['CLASS_NAMES'].split(',')
AZURE_CN = os.environ['AZURE_STORAGE_CONNECTION_STRING']
AZURE_CONTAINER = os.environ['AZURE_STORAGE_CONTAINER']
AZURE_TABLE = os.environ['AZURE_STORAGE_KEY_TABLE']
AZURE_PARTITION = os.environ['AZURE_STORAGE_KEY_PARTITION']

blob_service_client = BlobServiceClient.from_connection_string(AZURE_CN)
table_service = TableService(connection_string=AZURE_CN)

try:
    container_client = blob_service_client.create_container(AZURE_CONTAINER)
except: pass

if not os.path.isfile('./weights/custom.tf.index'): # if tensorflow file is not present download it 
    if not os.path.isfile('./weights/custom.zip'): 
        FILE = os.environ['WEIGHTS_FILE']
        urllib.request.urlretrieve(FILE, './weights/custom.zip')
    with zipfile.ZipFile('./weights/custom.zip', 'r') as zip_ref:
        zip_ref.extractall('./weights/')

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0: tf.config.experimental.set_memory_growth(physical_devices[0], True)
yolo = YoloV3(classes= len(CLASS_NAMES))
yolo.load_weights('./weights/custom.tf').expect_partial()

if os.name == 'nt': 
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello(): return random.choice([':)', ':|', ':(']), 200

@app.route('/predict', methods=['POST'])
def predict():

    if 'key' not in request.headers: abort(404)
    key = request.headers['key']

    try:
        owner = table_service.get_entity(AZURE_TABLE, AZURE_PARTITION, key)
    except:
        abort(401)
    
    rand = "".join(random.sample(string.ascii_lowercase + string.digits, 25)) # generate request id
    images = request.files.getlist("screens")
    if len(images) != 1: abort(404)

    image = images[0]
    image_ext =  os.path.splitext(image.filename)[1]
    image_name = rand + "_in" + image_ext
    image_temp = os.path.join(os.getcwd(), 'temp', image_name)
    image.save(image_temp)

    try:
        image_blob = key + '/' + image_name
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER, blob=image_blob)
        with open(image_temp, "rb") as data: blob_client.upload_blob(data)
    except: pass
    
    raw_img = tf.image.decode_image(open(image_temp, 'rb').read(), channels=3)
    np_img = raw_img.numpy()
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    img = tf.expand_dims(raw_img, 0)
    img = transform_images(img, IMAGE_SIZE)

    start = time.time()
    boxes, scores, classes, nums = yolo(img)
    detect = time.time() - start

    ocr_active = request.headers['ocr'].lower() == 'true' if 'ocr' in request.headers else False
    ocr_psm = int(request.headers['ocr-psm']) if 'ocr-psm' in request.headers else 7
    ocr_invert = request.headers['invert'].lower() == 'true' if 'invert' in request.headers else False
    ocr_config = f'--psm {ocr_psm}'

    response = []
    start = time.time()

    for i in range(nums[0]):

        detection = CLASS_NAMES[int(classes[0][i])]
        score = np.array(scores[0][i])
        box = np.array(boxes[0][i])
        height, width, channels = cv_img.shape
        confidence = float("{0:.2f}".format(score * 100))

        x = int(float(box[0]) * width)
        y = int(float(box[1]) * height)
        r = int(float(box[2]) * width)
        l = int(float(box[3]) * height)
        w = r - x
        h = l - y

        quadrants = [] # TODO: calculate quadrants of the detection on the image

        url = f'{request.host_url}detection/{key}/{rand}/{i}'
        if 'proxy' in request.headers:
            proxy = request.headers['proxy'].strip('/')
            url = f'{proxy}/detection/{key}/{rand}/{i}'

        current = { "confidence": confidence, "left": x, "top": y, "width": w, "height": h, "right": r, "bottom": l, "quadrants": quadrants, "url": url, "image_width": width, "image_height": height }

        if ocr_active: 
            
            crop_img = cv_img[y:l,x:r] # crop image on the detection box
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) # convert to grayscale
            
            if ocr_invert: # simple image thresholding
                crop_img = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
            
            try:
                crop_temp = os.path.join(os.getcwd(), 'temp', f'{rand}_det_{i}.png')
                cv2.imwrite(crop_temp, crop_img)
                crop_blob = blob_service_client.get_blob_client(container=AZURE_CONTAINER, blob=f'{key}/{rand}_det_{i}.png')
                with open(crop_temp, "rb") as data: crop_blob.upload_blob(data)
                os.remove(crop_temp)
            except: pass
                
            text = pytesseract.image_to_string(crop_img, config=ocr_config) # perform OCR
            text = re.sub(r"(?:[\s])+", " ", text, 0, re.MULTILINE).strip() # remove multiple spaces and trim ?????????
            current["text"] = text
        
        response.append(current)

    process = time.time() - start

    result = { "id": rand, "detections": response, "detect": detect, "process": process }

    try:
        stream = io.BytesIO()
        stream.write(str(result).encode('utf-8'))
        stream.seek(0)
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER, blob=f'{key}/{rand}.json')
        blob_client.upload_blob(stream)
    except: pass

    try:
        os.remove(image_temp)
    except: pass

    return jsonify(result), 200

@app.route('/detection/<string:key>/<string:id>/<int:index>')
def download(key, id, index):
    try:
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER, blob=f'{key}/{id}_det_{index}.png')
        streamdownloader=blob_client.download_blob()
        stream = io.BytesIO()
        streamdownloader.download_to_stream(stream)
        stream.seek(0)
        return send_file(stream, mimetype='image/png')
    except: pass
    abort(404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
