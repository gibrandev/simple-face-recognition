import face_recognition
known_image = face_recognition.load_image_file("dataset/gibran/00000.jpg")
sent_image = face_recognition.load_image_file("dataset/gibran/00001.jpg")

encoding = face_recognition.face_encodings(known_image)
sent_encoding = face_recognition.face_encodings(sent_image)

if len(sent_encoding) > 0:
    encoding = sent_encoding[0]
else:
    print(False)
    quit()

results = face_recognition.compare_faces(encoding, sent_encoding)
print(results)