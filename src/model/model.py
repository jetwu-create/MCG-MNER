from model.base_model import BaseModel
from model.modeling_t5 import T5ForConditionalGeneration
import clip
from pathlib import Path


class Mymodel(BaseModel):
    def __init__(self, args, model, device):
        self.args = args
        super(Mymodel, self).__init__(args)
        self.model = model
        self.to(device)

    @classmethod
    def from_pretrained(cls, args, model_path, device):
        v_path = f'./pretrain_model/ViT-B-32.pt'
        v_path = Path(v_path)
        v_encoder, _ = clip.load(v_path, device=device)
        t5_model = T5ForConditionalGeneration.from_pretrained(model_path, v_encoder)
        return cls(
            args=args,
            model=t5_model,
            device=device
        )


    def forward(self, input_ids, attention_mask, visuals_global, visuals_local, global_weights, local_weights, labels):
        output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            visuals_global=visuals_global,
                            visuals_local=visuals_local,
                            global_weights=global_weights,
                            local_weights=local_weights,
                            labels=labels,
                            )
    
        return output


    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)


    def generate(self, input_ids, attention_mask, visuals_global, visuals_local, global_weights, local_weights, **generation_kwargs):
        predict_ids = self.model.generate(
                                        input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        visuals_global=visuals_global,
                                        visuals_local=visuals_local,
                                        global_weights=global_weights,
                                        local_weights=local_weights,
                                        **generation_kwargs,
                                        )

        return predict_ids