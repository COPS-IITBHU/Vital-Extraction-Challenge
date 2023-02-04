from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction
)

def get_boxes(image):
  # image = read_image(img_path)
  refine_net = load_refinenet_model(cuda= False)
  craft_net = load_craftnet_model(cuda=False)
  prediction_result = get_prediction(
      image=image,
      craft_net=craft_net,
      refine_net=refine_net,
      text_threshold=0.7,
      link_threshold=0.4,
      low_text=0.4,
      cuda=False,
      long_size=640
  )
  return prediction_result['boxes']
