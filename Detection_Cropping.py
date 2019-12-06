import os
import glob
import dlib
from skimage import io
import openface
import tensorflow as tf
predictor_model = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()                                    # Create a HOG face detector using the built-in dlib class
face_pose_predictor = dlib.shape_predictor(predictor_model)                         # landmark 검출
face_aligner = openface.AlignDlib(predictor_model)                                  # alignDlib클래스 init함수를 보면 애초에 shape_predictor을 통해 landmark를 계산한다
                                                                                    # 그 후 affinetransform을 통해 중앙으로 landmark를 rotate, 및 scale 계산

def expand_img(image) :
    sess=tf.Session()
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image,max_delta=0.3)
    image = tf.image.random_contrast(image,lower=0.9,upper=1.1)
    image = tf.image.random_hue(image,max_delta=0.03)
    image = tf.image.random_saturation(image,lower=0.9,upper=1.1)

    return image.eval(session=sess)

def cropping_img(img_class) :
    input_path = os.getcwd()+'/image/'+img_class                                                      # 현재 디렉토리 위치를 불러온다
    output_path=os.getcwd()+'/aligned_img/'+img_class
    if not os.path.isdir(output_path) :
        os.mkdir(output_path)
    count=0

    for file_name in glob.glob(os.path.join(input_path, '*.jpg')):                 # path 지정한 곳에서 txt파일 모두뒤짐

        win = dlib.image_window()
        image = io.imread(file_name)
        detected_faces = face_detector(image, 1)

        print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

        win.set_image(image)

        for i, face_rect in enumerate(detected_faces):
            r=0
            print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                                     face_rect.right(), face_rect.bottom()))
            win.add_overlay(face_rect)

            pose_landmarks = face_pose_predictor(image, face_rect)
            alignedFace = face_aligner.align(128, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            win.add_overlay(pose_landmarks)
            io.imsave(output_path + '/' + img_class + "{}_{}.jpg".format(count, r), alignedFace)
            for r in range(1, 6) :
                rand_img=expand_img(alignedFace)
                io.imsave(output_path + '/' + img_class+"{}_{}.jpg".format(count, r), rand_img)

        count+=1
        #dlib.hit_enter_to_continue()

cropping_img('Parent')
cropping_img('Ganghee')
cropping_img('test_Gang')