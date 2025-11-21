import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform
import glob
from collections import deque

def load_and_prepare_images(image_paths, transform_fn):

    images = []
    labels = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        label_path = image_path.replace('images', 'masks').replace('.jpg', '.jpg')
        label = Image.open(label_path).convert('L')

        images.append(transform_fn(image))
        labels.append(transform_fn(label))
    images = torch.cat([tensorize_image(img).to(device) for img in images], dim=0)
    labels = torch.cat([tensorize_image(label, normalize=False).to(device) for label in labels], dim=0)

    return images, labels


def load_in_batches_with_deque(arr, batch_size):
    result = []
    window = deque(maxlen=batch_size)  
    for item in arr:
        window.append(item) 
        if len(window) == batch_size:
            result.append(list(window))
    return result

def tensorize_image(image, normalize=True):
    tensor = transforms.ToTensor()(image)
    if normalize:
        tensor = transforms.Normalize([0.5], [0.5])(tensor)
    return tensor.unsqueeze(0)
 
def save_output_images(output_tensors, input_paths, output_dir, view_length, batch_idx, batch_length):
    os.makedirs(output_dir, exist_ok=True)

    # Preload input image sizes to avoid repeated disk I/O
    input_sizes = [Image.open(p).size for p in input_paths]

    for j, output_tensor in enumerate(output_tensors):

        output_pil = transforms.ToPILImage()(output_tensor.cpu() * 0.5 + 0.5)
        frame = j // view_length
        view  = j % view_length
        idx   = frame * view_length + view

        save_flag = (
            (batch_idx == 0 and frame != 2) or # first batch, not last frame
            (batch_idx == batch_length - 1 and frame != 0) or # last batch, not first frame
            (batch_idx not in (0, batch_length - 1) and frame == 1) # middle batches, only middle frame
        )

        if not save_flag:
            continue

        output_pil = output_pil.resize(input_sizes[idx], Image.LANCZOS)
        # save
        out_name = os.path.basename(input_paths[idx])
        output_pil.save(os.path.join(output_dir, out_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=False, help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--weather_type', type=str, default=None, help='support foggy, rainy, snowy')
    parser.add_argument('--alpha', type=int, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--dataset', type=str, default='custom', help='the dataset to be used')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    parser.add_argument('--tv_att', type=bool, default=True,  help='Use temporal view attention')
    parser.add_argument('--with_seg', type=bool, default=True, help='Use segmentation maps as additional input')

    args = parser.parse_args()

    # --- model path ---
    model_path = "./ckpts/with_seg.pkl" 
    prompts_list = {
        "snowy": "Picture of a snowy weather scene",
        "foggy": "Picture of a foggy weather scene",
        "rainy": "Picture of a rainy weather scene",
    }
    prompt = prompts_list.get(args.weather_type, "")

    print(f"Using prompt: {prompt}")
    print(f"Using dataset: {args.dataset}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- dataset → frame/view settings ---
    dataset_settings = {
        "kitti":     {"frame": 3, "view": 2},
        "pandaset":  {"frame": 3, "view": 3},
        "nuscenes":  {"frame": 3, "view": 3},
        "waymo":     {"frame": 3, "view": 5},
        "custom":    {"frame": 1, "view": 1},
    }

    frame_length = dataset_settings.get(args.dataset, {"frame": 3})["frame"]
    view_length  = dataset_settings.get(args.dataset, {"view": 1})["view"]
    # disable tv_att for single-frame mode
    if args.dataset == "custom":
        args.tv_att = False
    args.tv_att = False
    # --- transforms & dataset paths ---
    T_val = build_transform(args.image_prep)
    image_root = f"./dataset/{args.dataset}/images"

    # --- collect frame IDs ---
    input_images = glob.glob(f"{image_root}/*.jpg")
    frames = sorted({os.path.basename(img).split("_")[0] for img in input_images})

    batches = load_in_batches_with_deque(frames,frame_length)
    batch_length = len(batches)

    model = CycleGAN_Turbo(frame_length,view_length,weather_type=args.weather_type, with_seg=args.with_seg, tv_att=args.tv_att, pretrained_path=model_path)
    model.eval()
    if args.use_fp16:
        model.half()
    for i, batch in enumerate(batches):

        batch_idx = i
        input_image_paths = [os.path.join(image_root, f'{frame}_{view}.jpg') for frame in batch for view in range(view_length)]

        input_length = frame_length*view_length
        x_t, x_t_label = load_and_prepare_images(input_image_paths, T_val)

        if args.use_fp16:
            x_t = x_t.half()
        with torch.no_grad():
            output = model(x_t, x_t_label,args.alpha,input_length, direction='a2b', caption=prompt, mode='inference',with_seg=args.with_seg)

        save_output_images(output, input_image_paths, args.output_dir,view_length,batch_idx,batch_length)

