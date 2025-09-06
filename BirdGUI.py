#####     Bird GUI     #####
import sys
import csv
import torch
import torchvision.transforms as transforms
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
    QFileDialog, QHBoxLayout, QVBoxLayout, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QFont

import _BirdNet

### ----- 模型加载与预处理 ----- ###
species_file = "./species_list.csv"
class BirdClassifier():
    def __init__(self, version:str):
        """Bird classifier."""
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

    def classify(self, image:torch.Tensor) -> list:
        """Image tensor -> top3 predictions."""
        with torch.no_grad():
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

### ----- GUI线程 ----- ###
class GUI_Thread(QThread):
    prediction_done = pyqtSignal(list)

    def __init__(self, classifier:BirdClassifier, image_path):
        super().__init__()
        self.classifier = classifier
        self.image_path = image_path

    def run(self):
        try:
            tensor = self.classifier.preprocess(self.image_path)
            results = self.classifier.classify(tensor)
            self.prediction_done.emit(results)
        except Exception as e:
            self.prediction_done.emit([("Error", str(e))])

### ----- 主界面 ----- ###
class BirdGUI(QMainWindow):
    def __init__(self, classifier:BirdClassifier):
        """GUI Window for Bird Classifier."""
        super().__init__()
        self.classifier = classifier
        self.init_ui()
        self.setAcceptDrops(True)

    def init_ui(self):
        """Initialize the GUI layout."""
        ### 窗口设置
        self.setWindowTitle("Bird Classifier")
        self.setGeometry(400, 400, 1600, 1000)

        ### 字体设置
        font = QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        QApplication.setFont(font)

        ### 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        ### 左侧图像显示
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa;")
        layout.addWidget(self.image_label, stretch=60)

        ### 右侧布局
        right_layout = QVBoxLayout()
        layout.addLayout(right_layout, stretch=40)
        # 状态显示区
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        right_layout.addWidget(self.status_text, stretch=40)
        # 结果显示区
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setHtml("<style>body {line-height: 1.5em;}</style>")
        right_layout.addWidget(self.result_text, stretch=60)

        ### 菜单栏
        menubar = self.menuBar()
        file_menu = menubar.addMenu(" File ")
        file_menu.addAction("Open", self.open_file)
        file_menu.addAction("Clear", self.clear)
        help_menu = menubar.addMenu(" Help ")
        help_menu.addAction("Help EN", self.show_helpEN)
        help_menu.addAction("Help CN", self.show_helpCN)
        help_menu.addAction("About", self.show_info)

    ### ---------- 拖放支持 ----------
    def dragEnterEvent(self, event:QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event:QDropEvent):
        url = event.mimeData().urls()[0]
        self.process_image(url.toLocalFile())

    ### ---------- 功能逻辑 ----------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self,
            caption="选择图像", filter="图像文件 (*.jpg *.jpeg *.png)")
        if path: self.process_image(path)

    def clear(self):
        self.image_label.clear()
        self.status_text.clear()
        self.result_text.clear()
    
    def show_helpCN(self):
        helpText = """这是一个简单的鸟类图像识别工具。\n
        1. 点击`File`-`Open`打开图像文件。图像会显示在左侧，识别结果会显示在右侧。
        2. 点击`File`-`Clear`清除图像和识别结果。
        3. 点击`Help`-`Help CN`查看帮助。
        4. 点击`Help`-`About`查看项目信息。
        """
        self.status_text.setText(helpText)
    
    def show_helpEN(self):
        helpText = """This is a simple bird image classification tool.\n
        1. Click `File`-`Open` to open an image file. The image will be displayed on the left, and the results on the right.
        2. Click `File`-`Clear` to clear the image and classification results.
        3. Click `Help`-`Help EN` to view the help.
        4. Click `Help`-`About` to view the project information.
        """
        self.status_text.setText(helpText)
    
    def show_info(self):
        info = "Bird Classifier (Ver 1)\n\n" +\
            "https://github.com/Firefly-ZJ/Bird-Classification"
        self.status_text.setText(info)

    ### ---------- 图像分类 ----------
    def process_image(self, image_path):
        """Show the image and run classification."""
        # 显示图像
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(768, 768, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

        # 启动预测线程
        self.status_text.setText("Classifying...\n正在识别中...")
        self.thread = GUI_Thread(self.classifier, image_path)
        self.thread.prediction_done.connect(self.show_results)
        self.thread.start()

    def show_results(self, results):
        if results[0][0] == "Error":
            QMessageBox.critical(self, "错误", results[0][1])
            return

        # 显示结果
        output = []
        for _, (label, prob, (latin, chinese)) in enumerate(results):
            if latin == "Unknown":
                output.append(f"未识别到有效物种<br>")
            else:
                output.append(f"<b><i>{latin}</i></b><br>"
                              f"{chinese}  （{prob*100:.2f} %）<br>")
        
        self.result_text.setHtml("<hr>".join(output))
        self.status_text.setText("Completed\n识别完成")

### ----- 主函数 ----- ###
if __name__ == "__main__":
    model_version = "v1large"
    classifier = BirdClassifier(model_version)
    print(classifier)

    app = QApplication(sys.argv)
    window = BirdGUI(classifier)
    window.show()
    sys.exit(app.exec_())