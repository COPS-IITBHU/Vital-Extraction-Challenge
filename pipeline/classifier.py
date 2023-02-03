import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F


class Classifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = None
        self.device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
        self.tensor = transforms.ToTensor()
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    
    def loadResnet18Classifier(self, chkpt_path):
        self.model = models.resnet18()
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        checkpoint = torch.load(chkpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        return

    def predict(self, x):
        model.eval()
        fsize = list(x.size())
        if len(fsize) == 3:
            fsize = [1]+fsize
        
        x = self.tensor(x)
        resize = transforms.Resize(fsize)
        x = resize(x)
        x = self.norm(x)
        x = x.to(self.device)
        outputs = self.model(x)
        _, preds = torch.max(F.softmax(outputs, dim = 1), 1)

        return preds.numpy()[0]