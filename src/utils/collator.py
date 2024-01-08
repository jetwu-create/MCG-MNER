import torch
from torchvision import transforms
import os
from PIL import Image


class Collator:
    def __init__(
        self,
        tokenizer,
        visual_global_path,
        visual_local_path,
        visual_global_weight,
        visual_local_weight,
        tokenizer_kwargs,
    ):
        self.tokenizer = tokenizer
        self.visual_global_weight = visual_global_weight
        self.visual_local_weight = visual_local_weight
        self.tokenizer_kwargs = tokenizer_kwargs
        self.visual_global_path = visual_global_path
        self.visual_local_path = visual_local_path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __call__(self, batch):
        
        context_list, question_list, answer_list = [], [], []
        global_weights, local_weights = [], []
        visuals_global, visuals_local = [], []
        for i in range(len(batch)):
            instance = batch[i]
            context_list.append(instance.context)
            question_list.append(instance.question)
            answer_list.append(instance.answer)
            index = instance.img_idx
            if index in self.visual_global_weight.keys() and index in self.visual_local_weight.keys():
                weights_temp1 = self.visual_global_weight[index]
                weights_temp2 = self.visual_local_weight[index]
                global_weights.append(weights_temp1)
                local_weights.append(weights_temp2)
                
            v_global_path = self.visual_global_path / index
            v_local_path = self.visual_local_path / index
            
            if not os.path.exists(v_local_path):
                pass
            elif not os.path.exists(v_global_path):
                 print(v_global_path)
            try:
                visual_global, visual_local = Image.open(v_global_path).convert('RGB'), Image.open(v_local_path).convert('RGB')
                visual_global, visual_local = self.transform(visual_global), self.transform(visual_local)
                visuals_global.append(visual_global)
                visuals_local.append(visual_local)
            except:
                v_global_path = self.visual_global_path / "17_06_4705.jpg"
                v_local_path = self.visual_local_path / "17_06_4705.jpg"
                visual_global, visual_local = Image.open(v_global_path).convert('RGB'), Image.open(v_local_path).convert('RGB')
                visual_global, visual_local = self.transform(visual_global), self.transform(visual_local)
                visuals_global.append(visual_global)
                visuals_local.append(visual_local)
                
        visuals_global = torch.stack([item for item in visuals_global])
        visuals_local = torch.stack([item for item in visuals_local])
        global_weights = torch.tensor(global_weights).view(len(batch), 1, 1)
        local_weights = torch.tensor(local_weights).view(len(batch), 1, 1)
        
        tokenized_batch = self.tokenizer(
            context_list, question_list, **self.tokenizer_kwargs
        )
        
        with self.tokenizer.as_target_tokenizer():
            answers = self.tokenizer(answer_list, **self.tokenizer_kwargs)
            
        tokenized_batch["answers"] = answers
        tokenized_batch["instances"] = batch
        tokenized_batch["global_weights"] = global_weights
        tokenized_batch["local_weights"] = local_weights
        tokenized_batch["visuals_global"] = visuals_global
        tokenized_batch["visuals_local"] = visuals_local
        
        return tokenized_batch