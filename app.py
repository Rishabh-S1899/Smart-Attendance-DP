from flask import Flask, request, jsonify
# from mtcnn import MTCNN
import cv2
import numpy as np
import base64
import tempfile
app = Flask(__name__)
# from transformers import AutoImageProcessor, EfficientNetModel
# import torch
import cv2
# from mtcnn import MTCNN
import warnings
import numpy as np 
import os 
import face_recognition
from PIL import Image
import io
warnings.filterwarnings("ignore")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
# model = EfficientNetModel.from_pretrained("google/efficientnet-b7", output_hidden_states=True).to(device)
# detector = MTCNN()


@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        # Get the video from the request
        video_base64 = request.json['video']
        video_buffer = np.frombuffer(base64.b64decode(video_base64), dtype=np.uint8)

        # Create a temporary file for the video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_video.write(video_buffer)
            temp_video_path = temp_video.name

        # Read the video
        video_capture = cv2.VideoCapture(temp_video_path)

        all_frame_features = []

        while True:
            # Read a frame from the video
            ret, frame = video_capture.read()
            if not ret:
                # No more frames left
                break

            # Perform face detection and encoding on each frame
            face_locations = face_recognition.face_locations(frame)
            all_face_encodings = face_recognition.face_encodings(frame, face_locations)
            print(len(all_face_encodings))

            for face_location, face_encoding in zip(face_locations, all_face_encodings):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue())
                all_frame_features.append([img_str.decode('utf-8'), face_encoding.tolist()])

        # Release the video capture object
        video_capture.release()

        # Remove the temporary video file
        os.remove(temp_video_path)
        print(len(all_frame_features))

        return jsonify(all_frame_features)

    except Exception as e:
        return jsonify({'error': str(e)})

        #     faces = detector.detect_faces(frame)
        #     face_list = []
        #     for i, face in enumerate(faces):
        #         x, y, width, height = face['box']
        #         face_image = frame[y:y+height, x:x+width]
        #         cv2.imwrite(f'face_{i}_{k}.jpg', face_image)
        #         face_list.append(f'face_{i}_{k}.jpg')
        #     face_feature = {}
        #     for i in face_list:
        #         img = str(i)
        #         image = cv2.imread(img)
        #         _, encoded_image = cv2.imencode('.jpg', image)
        #         encoded_image_base64 = base64.b64encode(encoded_image).decode('utf-8')
        #         inputs = image_processor(image, return_tensors="pt").to(device)
        #         with torch.no_grad():
        #             outputs = model(**inputs)
        #             last_hidden_states = outputs.pooler_output
        #             print(f"last hidden states shape is {last_hidden_states.shape}")
        #             lsh = last_hidden_states.cpu().tolist()
        #             lsh = lsh[0]
        #             face_feature[encoded_image_base64] = lsh
        #     list_of_lists = []
        #     for key, value in face_feature.items():
        #         sublist = [key, value]
        #         list_of_lists.append(sublist)
        #     print(f"frame {k} done")
        #     all_frame_features.extend(list_of_lists)
        # # Release the video capture object
        # video_capture.release()
        # os.unlink(temp_video_path)
        # print(f'This is shape of 0th {len(all_frame_features[0])}')
        # print(f'This is shape {len(all_frame_features)}')
        # return jsonify(all_frame_features)
        # return "Hi"

    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
                # image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
                # model = EfficientNetModel.from_pretrained("google/efficientnet-b7",output_hidden_states=True)
                # detector = MTCNN()
        image_base64 = request.json['image']
        image_buffer = np.frombuffer(base64.b64decode(image_base64), dtype=np.uint8)
        image = cv2.imdecode(image_buffer, flags=cv2.IMREAD_COLOR)
        cv2.imwrite('temp.jpg', image)
        images = face_recognition.load_image_file("temp.jpg")
        face_locations = face_recognition.face_locations(images)
        all_face_encodings = face_recognition.face_encodings(images)
        print(f'This is the length of all_face_encodings {len(all_face_encodings)}')
        result_vector = []

        for idx, face_location in enumerate(face_locations):

            # Print the location of each face in this image
            top, right, bottom, left = face_location
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

            # You can access the actual face itself like this:
            face_image = images[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            buffered = io.BytesIO()

            # Save the PIL Image to the BytesIO object in JPEG format
            pil_image.save(buffered, format="JPEG")

            # Get the value of the BytesIO buffer
            img_str = base64.b64encode(buffered.getvalue())

            # Append the base64 encoded image to the list
            result_vector.append([img_str.decode('utf-8'),all_face_encodings[idx].tolist()])

        return jsonify(result_vector)
    except Exception as e:
        return str(e)



@app.route('/student_upload', methods=['POST'])
def student_upload_image():
    try:
        image_base64 = request.json['image']
        image_buffer = np.frombuffer(base64.b64decode(image_base64), dtype=np.uint8)
        image = cv2.imdecode(image_buffer, flags=cv2.IMREAD_COLOR)
        cv2.imwrite('temp.jpg', image)
        images = face_recognition.load_image_file("temp.jpg")   
        # face_locations = face_recognition.face_locations(images)
        all_face_encodings=face_recognition.face_encodings(images)
        print(len(all_face_encodings))
        embed=all_face_encodings[0].tolist()

        return embed
        # image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
        # model = EfficientNetModel.from_pretrained("google/efficientnet-b7",output_hidden_states=True)
        # detector = MTCNN()
        # image_base64 = request.json['image']
        # image_buffer = np.frombuffer(base64.b64decode(image_base64), dtype=np.uint8)
        # image = cv2.imdecode(image_buffer, flags=cv2.IMREAD_COLOR)
        # print("hi")
        # faces = detector.detect_faces(image)

        # for i,face in enumerate(faces):
        #     x, y, width, height = face['box']
        #     face_image = image[y:y+height, x:x+width]

        # print("yo")
        # _, encoded_image = cv2.imencode('.jpg', face_image)
        # encoded_image_base64 = base64.b64encode(encoded_image).decode('utf-8')
        # inputs = image_processor(face_image, return_tensors="pt")
        # input_img=inputs
        # with torch.no_grad():
        #     outputs = model(**input_img)
        # last_hidden_states = outputs.pooler_output
        # print(last_hidden_states.shape)
        # lsh=last_hidden_states.tolist()
        # lsh = lsh[0]
        # # face_feature[encoded_image_base64]=lsh 

        # print(lsh)
        # # lsh_string=''.join(map(str, lsh))
        # return lsh
        # return "hi"
    except Exception as e:
        print("Error has been executed")
        return str(e), 400
    

if __name__ == '__main__':
    app.run(debug=False)

