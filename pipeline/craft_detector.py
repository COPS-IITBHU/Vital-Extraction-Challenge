from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

def get_boxes(image):
  # image = read_image(img_path)
  refine_net = load_refinenet_model(cuda=True)
  craft_net = load_craftnet_model(cuda=True)
  prediction_result = get_prediction(
      image=image,
      craft_net=craft_net,
      refine_net=refine_net,
      text_threshold=0.7,
      link_threshold=0.4,
      low_text=0.4,
      cuda=True,
      long_size=1280
  )
  return prediction_result['boxes']
