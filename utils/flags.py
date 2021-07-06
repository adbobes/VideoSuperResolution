import argparse

parser = argparse.ArgumentParser()

# Main
parser.add_argument("-train_or_inference", "--train_or_inference", dest="train_or_inference", type=int, default=1,
                    help="Train or inference, 1 for train, 2 for inference.")

# Training details
parser.add_argument("-batch_size", "--batch_size", dest="batch_size", type=int, default=1,
                    help="Batch size of the input batch")
parser.add_argument("-epoch_num", "--epoch_num", dest = "epoch_num", type = int, default = 100000,
                    help = "The max epoch for the training")
parser.add_argument("-frames", "--frames", dest = "frames", type = int, default = 10,
                    help = "The number of the reccurent lenght")

# Training data settings
parser.add_argument("-LR_root_path", "--LR_root_path", dest="LR_root_path", type=str, default="dataset/LR",
                    help="Path of LR folder")
parser.add_argument("-root_video_dir", "--root_video_dir", dest = "root_video_dir", type = str, default = "dataset",
                    help = "The root directory of the dataset, it will containt scene + LR + HR folders")



# Optimizer
parser.add_argument("-lr", "--lr", dest="lr", type=float, default=0.0001,
                    help="The learning rate for the network")

# Checkpoint
parser.add_argument("-load_checkpoint", "--load_checkpoint", dest="load_checkpoint", type=bool, default=0,
                    help="Loading checkpoint, 1 true, 0 false.")
parser.add_argument("-save_checkpoint_path","--save_checkpoint_path",dest = "save_checkpoint_path", type = str,
                    default ="saves/", help = "Path for checkpoints folder to be saved." )
parser.add_argument("-checkpoint_file","--checkpoint_file",dest = "checkpoint_file", type = str,
                    default ="", help = "Checkpoint file name." )

# Utils

parser.add_argument("-crop_size", "--crop_size", dest = "crop_size", type = int, default = 32,
                    help = "The crop size of the training image")
parser.add_argument("-HR_root_path", "--HR_root_path", dest = "HR_root_path", type = str, default = "dataset/HR",
                    help = "Path of HR folder")
parser.add_argument("-input_video_dir",'--input_video_dir', dest = "input_video_dir", type = str, default = "dataset/scene",
                    help = "The directory of the video input data, for training")
parser.add_argument("-str_dir", "--str_dir", dest = "str_dir", type = int, default= 1,
                    help = "The starting index of the video directory")
parser.add_argument("-end_dir", "--end_dir", dest = "end_dir", type = int, default = 30,
                    help = "The ending index of the video directory")
parser.add_argument("-input_video_pre", "--input_video_pre", dest = "input_video_pre", type = str, default = "scene",
                    help = "The pre of the directory of the video input data")
parser.add_argument("-max_frm", "--max_frm", dest = "max_frm", type = int, default = 120,
                    help = "Should be duration number of frames from 'input_video_dir' directory.")

# Inference
parser.add_argument("-num_of_epochs", "--num_of_epochs", dest = "num_of_epochs", type = int, default = 80,
                    help = "Should be duration number of frames from 'input_video_dir' directory.")

parser.add_argument("-epoch_step", "--epoch_step", dest = "epochs_step", type = int, default = 40,
                    help = "Should be duration number of frames from 'input_video_dir' directory.")

parser.add_argument("-inference_path", "--inference_path", dest = "inference_path", type = str, default = "inference/input",
                    help = "Should be duration number of frames from 'input_video_dir' directory.")
                    
parser.add_argument("-start_epoch", "--start_epoch", dest = "start_epoch", type = int, default = 0,
                    help = "Should be duration number of frames from 'input_video_dir' directory.")

flags = parser.parse_args()
