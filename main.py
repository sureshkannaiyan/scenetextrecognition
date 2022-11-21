# Image text removal
import json

from PIL import Image
import keras_ocr, cv2, uuid, math, os
import numpy as np
from flask import Flask, request


RESULTS_FOLDER = "./results/"

app = Flask(__name__)
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

pipeline = keras_ocr.pipeline.Pipeline()

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


def get_ocr(img):
    img.save("img.png")
    # read image
    img = keras_ocr.tools.read("img.png")
    # generate (word, box) tuples
    prediction_groups = pipeline.recognize([img])
    res = {}
    for ocr in prediction_groups[0]:
        res[ocr[0]] = ocr[1].tolist()

    return json.dumps(res)


def image_text_removal(img_path, img_id):
    # read image
    img = keras_ocr.tools.read(img_path)
    # generate (word, box) tuples
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,
                 thickness)
        img = cv2.inpaint(img, mask, 11, cv2.INPAINT_NS)
        img = cv2.medianBlur(img, 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cv2.imwrite(RESULTS_FOLDER + img_id + "/" +img_id + "_OUTPUT.jpg", img)
    Image.fromarray(img).save(RESULTS_FOLDER + img_id + "/" +img_id + "_OUTPUT.jpg")
    return "success"


@app.route("/input", methods=['GET', 'POST'])
def image_text_removal_api():
    if request.method == 'POST':
        imagefile = request.files["image"]
        flag = request.form["flag"]
        if flag == "ocr only":
            processed_img = get_ocr(imagefile)
        else:
            if imagefile:
                img_id = str(uuid.uuid1())
                os.makedirs(RESULTS_FOLDER + img_id)
                img_path = RESULTS_FOLDER + img_id + "/" + img_id + ".jpg"
                imagefile.save(img_path)
                processed_img = get_ocr(img_path, img_id)
                # processed_img = image_text_removal(img_path, img_id)
            else:
                processed_img = "image not received"
    else:
        processed_img = "Request could not be processed"
    return processed_img


if __name__ == "__main__":
    app.run(debug=False)

