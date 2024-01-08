from torch.utils.data import Dataset
from tqdm import tqdm
from utils.datatypes import TaskType
from utils.formatters import EET_, ETT_, ETCT_


class T5MNERDataset(Dataset):
    def __init__(
        self,
        data,
        instructions,
        options,
        tasks=None,
    ):
        
        super().__init__()
        if tasks is None:
            tasks = (TaskType.EET, TaskType.ETT, TaskType.ETCT)
            
        self.instances = self._convert_list_to_instance(
            data=data,
            instructions=instructions,
            options=options,
            tasks=list(tasks),
        )
    

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index):
        return self.instances[index]
    
    def _convert_list_to_instance(
        self,
        data,
        instructions,
        options,
        tasks,
    ):
        instances = []
        for item in tqdm(data):
            instance = self._convert_item_to_instances(
                data_item=item,
                instructions=instructions,
                options=options,
                multi_tasks=tasks,
            )
            instances.extend(instance)
            
        return instances
        
    def _convert_item_to_instances(
        self,
        data_item,
        instructions,
        options,
        multi_tasks,
    ):
        instances_ = []
        task_to_formatter = {
            TaskType.ETCT: ETCT_(),
            TaskType.ETT: ETT_(),
            TaskType.EET: EET_(),
        }
        for task in multi_tasks:
            if task.value not in instructions:
                continue
            
            context = data_item['context']
            entity_values = data_item["entity_values"]
            entity_spans = data_item["entity_spans"]
            img_idx = data_item['image_index']
            img_idx = img_idx + '.jpg'
            
            instance_ = task_to_formatter[task].format_instance(
                context=context,
                entity_values=entity_values,
                entity_spans=entity_spans,
                img_idx=img_idx,
                instruction=instructions[task.value],
                options=options,
            )
            
            instances_.append(instance_)
            
        return instances_