import onnxruntime
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from PIL import Image
from decimal import Decimal, ROUND_DOWN

class PredictImg_corn():
    def onnxPredict(self,filename):
        providers = ['CPUExecutionProvider']
        ort_session = onnxruntime.InferenceSession('resnet18_corn.onnx', providers=providers)

        x = torch.randn(1, 3, 256, 256).numpy()

        # onnx runtime 输入
        ort_inputs = {'input': x}

        # 测试集图像预处理-RCTN缩放裁剪、转 Tensor、归一化
        test_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(256),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                                            ])

        # 用 pillow 载入
        img_pil = Image.open(filename)

        input_img = test_transform(img_pil)

        input_tensor = input_img.unsqueeze(0).numpy()

        # ONNX Runtime 输入
        ort_inputs = {'input': input_tensor}

        # ONNX Runtime 输出
        pred_logits = ort_session.run(['output'], ort_inputs)[0]
        pred_logits = torch.tensor(pred_logits)

        pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

        # 取置信度最大的 n 个结果
        n = 3

        top_n = torch.topk(pred_softmax, n)

        # 预测类别
        pred_ids = top_n.indices.numpy()[0]

        # 预测置信度
        confs = top_n.values.numpy()[0]

        # 载入类别和对应 ID
        idx_to_labels = np.load('idx_to_labels_corn.npy', allow_pickle=True).item()

        result = []
        for i in range(n):
            class_name = idx_to_labels[pred_ids[i]]  # 获取类别名称
            confidence = confs[i] * 100  # 获取置信度\
            confidence_four = "{:.8f}".format(confidence)
            result.append(class_name)
            result.append(confidence_four)
        return result
    
        
    
if __name__ == '__main__':
    op = PredictImg_corn()
    result = op.onnxPredict('testimg.jpg')
    print(result)