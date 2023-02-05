import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from Dataloaders import test_transform


class Classifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = None
        self.device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
    
    def loadResnet18Classifier(self, chkpt_path):
        self.model = models.resnet18()
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        checkpoint = torch.load(chkpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        return

    def predict(self, img):
        img = test_transform(image=img)["image"]
        
        ## RUNNING THROUGH MODEL
        img = img.to(self.device).unsqueeze(0)
        outputs = self.model(img)
        _, preds = torch.max(F.softmax(outputs, dim = 1), 1)

        return preds.cpu().numpy()[0]

co = Classifier(4)
co.loadResnet18Classifier("weights/resnet18_weights")    
def classification(img):
    return co.predict(img)
    
    