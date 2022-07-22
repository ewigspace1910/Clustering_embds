import os
import glob
import os.path as osp
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class ReID_Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(ReID_Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        fname, camid = self.dataset[indices]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, camid 

#unlable image folder
def _process_dir(dir_path):
    print("load images from ->{}".format(dir_path))
    folder_paths = [p.path for p in os.scandir(dir_path)]
    folder_paths = [dir_path] if len(folder_paths) == 0 else folder_paths 
    img_paths = []
    for x in folder_paths:
        img_paths += glob.glob(osp.join(x, '*.jpg'))
    #pattern = re.compile(r'([-\d]+)_c(\d)')
    dataset = []
    for img_path in img_paths:
        camid = int(img_path.split("/")[-2].split("_")[-1])  #/unlabeled_wcam_dataset/bounding_box_test/cam_1/00001.jpg
        dataset.append((img_path, camid))

    print("\t--> include : {} images".format(len(dataset)))
    return dataset



#refine DataLoader for other dataset type
def get_data(data_dir, height, width, batch_size, workers=2):
    root = osp.join(data_dir)
    dataset = _process_dir(root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    test_loader = DataLoader(
        ReID_Preprocessor(list(dataset),
                    root=None, transform=test_transformer),
                    batch_size=batch_size, num_workers=workers,
                    shuffle=False, pin_memory=True)

    return test_loader