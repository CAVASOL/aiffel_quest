# from tensorflow.keras.applications.vgg16 import preprocess_input
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.imagenet_utils import decode_predictions

# async def prediction_model():
#     model = tf.keras.models.load_model('./vgg16.h5')
    
#     img = Image.open('./sample_data/jellyfish.jpg')

#     target_size = 224
#     img = img.resize((target_size, target_size)) 

#     np_img = image.img_to_array(img)

#     img_batch = np.expand_dims(np_img, axis=0)

#     pre_processed = preprocess_input(img_batch)
    
#     y_preds = model.predict(pre_processed)
#     np.set_printoptions(suppress=True, precision=5)  

#     result = decode_predictions(y_preds, top=1)
#     result = {"predicted_label" : str(result[0][0][1]), "prediction_score" :  str(result[0][0][2])}
    
#     return result