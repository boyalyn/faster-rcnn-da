import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class JACSDatasetBase(Dataset):

    def __init__(self, root, anno_file, size=(512,512), domain_label="source"):

        self.root = root
        anno_dict = json.load(open(os.path.join(root,anno_file),'r'))

        img_list = anno_dict["images"]
        anno_list = anno_dict["annotations"]

        self.size = size

        # domain label
        self.domain_label = 0 if domain_label=="source" else 1

        data_dic = dict()

        for img_dict in img_list:
            key = img_dict["id"]
            # if key != 0:
            #     continue
            val = {"file_name": img_dict["file_name"],
                   "height": img_dict["height"],
                   "width": img_dict["width"],
                   "instances": []}
            data_dic[key] = val

        for anno_dict in anno_list:
            img_id = anno_dict["image_id"]
            # if img_id != 0:
            #     continue
            bbox = anno_dict["bbox"]
            cls = anno_dict["category_id"]
            cur_dict = {"bbox": bbox,
                        "cls": cls}
            data_dic[img_id]["instances"].append(cur_dict)

        self.data_dic = data_dic
        self.data_ids = list(data_dic.keys())
        
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])
        print(self.__len__())
        
    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, index):
        img, h, w = self.fetch_img(index)
        bboxs, labels = self.fetch_anno(index, h, w)
        return img, bboxs, labels, torch.tensor(1.)
    
    def fetch_img(self,idx):
        
        id = self.data_ids[idx]
        file_name = self.data_dic[id]["file_name"]
        img_path = os.path.join(self.root, "Images", file_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        height, width = self.data_dic[id]["height"], self.data_dic[id]["width"]
        return img, height, width
    
    def fetch_anno(self,idx, h, w):
        
        id = self.data_ids[idx]
        anno = self.data_dic[id]["instances"]
        bboxs, labels = [], []

        for instance in anno:
            x_min, y_min, o_width, o_height = instance['bbox']
            x_max = x_min + o_height
            y_max = y_min + o_width

            ori_h, ori_w = h, w
            tgt_h, tgt_w = self.size
            
            x_min, x_max = tgt_w*x_min/ori_w, tgt_w*x_max/ori_w
            y_min, y_max = tgt_h*y_min/ori_h, tgt_h*y_max/ori_h

            bboxs.append([y_min, x_min, y_max, x_max])
            labels.append(instance["cls"])

        bboxs = torch.tensor(bboxs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return bboxs, labels


if __name__ == "__main__":

    dataset = JACSDatasetBase(root="/Users/boyaliu/Projects/JACS/Database/data_FHPB/trainset",
                              anno_file="/Users/boyaliu/Projects/JACS/Database/data_FHPB/trainset/trainset.json")
    print(dataset.__len__())
    print(dataset.__getitem__(0)[2].shape)
    
    
        


        