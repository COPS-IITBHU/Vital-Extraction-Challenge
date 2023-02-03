from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
DEVICE = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"

model = model.to(DEVICE)

def trOCR(img, locs):
    x, y, w, h = locs[0], locs[1], locs[2], locs[3]

    x *= img.shape[1]
    w *= img.shape[1]
    y *= img.shape[0]
    h *= img.shape[0]

    crop = img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    pixel_values = processor(crop, return_tensors="pt").pixel_values
    pixel_values = torch.tensor(pixel_values).to(DEVICE)
    
    generated_ids = model.generate(pixel_values)
    generated_text2 = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text2