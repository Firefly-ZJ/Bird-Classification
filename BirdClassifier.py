#####     Bird GUI     #####
import csv
import torch
import torchvision.transforms as transforms
from PIL import Image

import _BirdNet

species_file = "./species_list.csv"

class BirdClassifier():
    def __init__(self, version:str):
        """Bird classifier. Input the image path, and get the top3 predictions."""
        self.version = version
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "xpu" if torch.xpu.is_available() else "cpu")
        self.model = _BirdNet.getModel(version, load_weight=True)
        self.model.to(self.device).eval()
        
        self.speciesNames = []
        with open(species_file, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader: self.speciesNames.append((row[0], row[1]))
            self.speciesNames.pop(0) # 去除表头
        self.unknown = ("Unknown", "Unknown")

    def preprocess(self, image_path):
        """Image preprocessing."""
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def classify(self, image:torch.Tensor) -> list:
        """Image tensor -> top3 predictions."""
        outputs = self.model(image)
        probs = torch.softmax(outputs, dim=1)[0]
        top3_probs, top3_indices = torch.topk(probs, k=3)

        result = [(idx.item(), prob.item(), self._get_species_name(idx.item()))
            for idx, prob in zip(top3_indices, top3_probs)]
        return result

    def _get_species_name(self, label:int):
        if label < len(self.speciesNames):
            return self.speciesNames[label]
        else:
            return self.unknown
    
    def __str__(self):
        text = "Bird Classifier\n" +\
            f"Version: {self.version},  Device: {self.device}"
        return text

if __name__ == "__main__":
    model_version = "v1large"
    classifier = BirdClassifier(model_version)
    print(classifier)
    
    image_path = "./birdData/Ps_245167185.png"  # example image
    #image_path = "PATH/OF/YOUR/IMAGE"
    image = classifier.preprocess(image_path)
    predictions = classifier.classify(image)
    print("\nClassification Result:")
    for result in predictions:  # classification result
        print(result)
