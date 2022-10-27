from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from util.getMap import *
import torchvision.transforms as transforms

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))
        import torchvision.transforms as transforms
        self.transform_map = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5,), (0.5,))])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        trans_size = transforms.Resize([A.shape[1],A.shape[2]], Image.BICUBIC) # 原图测试时如果有resize，尺寸会变为4的倍数，此时map也要相应改变
        A_img = trans_size(A_img)
        A_map = self.transform_map(getMap(A_img, self.opt.self_attention_thresh))
        return {'A': A, 'A_paths': A_path, 'A_map':A_map}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
