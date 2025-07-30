from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

app = Flask(__name__, static_folder="uploads")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained model
from model import CustomCNN
model = CustomCNN()
model.load_state_dict(torch.load("cat_dog_model.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            image_url = f"/{filepath}"

            img = Image.open(filepath).convert("RGB")
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(img)
                pred = torch.argmax(output, dim=1).item()
                prediction = "Cat" if pred == 0 else " Dog"

    return render_template("index.html", prediction=prediction, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
