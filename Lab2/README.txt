## Environment Setup
pip install -r requirements.txt

## Training

# UNet
python src/train.py --model unet --data_root dataset/oxford-iiit-pet --splits_dir <splits_dir> --epochs 50 --batch_size 16

# ResNet34-UNet
python src/train.py --model resnet34_unet --data_root dataset/oxford-iiit-pet --splits_dir <splits_dir> --epochs 130 --batch_size 16

Trained model weights will be saved to saved_models/unet_best.pth and saved_models/resnet34_unet_best.pth.

## Evaluate (Dice Score on Validation Set)

# UNet
python src/evaluate.py --model unet --data_root dataset/oxford-iiit-pet --checkpoint saved_models/unet_best.pth --splits_dir <splits_dir> --scan_threshold

# ResNet34-UNet
python src/evaluate.py --model resnet34_unet --data_root dataset/oxford-iiit-pet --checkpoint saved_models/resnet34_unet_best.pth --splits_dir <splits_dir> --scan_threshold


## Inference

# UNet
python src/inference.py --model unet --data_root dataset/oxford-iiit-pet --checkpoint saved_models/unet_best.pth --test_list <test_list> --output unet_pred.csv --threshold 0.5 --tta

# ResNet34-UNet
python src/inference.py --model resnet34_unet --data_root dataset/oxford-iiit-pet --checkpoint saved_models/resnet34_unet_best.pth --test_list <test_list> --output resnet34_unet_pred.csv --threshold 0.5 --tta

<splits_dir>  : path to the directory containing train.txt and val.txt (provided by Kaggle)
<test_list>   : path to the test image list txt file (provided by Kaggle)
