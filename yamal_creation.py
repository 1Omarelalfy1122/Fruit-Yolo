import yaml
import os

def create_data_yaml(path_to_classes_txt, path_to_data_yaml, dataset_root):
    # Read classes.txt to get class names
    if not os.path.exists(path_to_classes_txt):
        print(f'classes.txt file not found! Please create it at {path_to_classes_txt}')
        return
    with open(path_to_classes_txt, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    number_of_classes = len(classes)

    # Create data dictionary
    data = {
        'path': dataset_root,
        'train': 'images/train',
        'val': 'images/val',
        'nc': number_of_classes,
        'names': classes
    }

    # Write data.yaml
    with open(path_to_data_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f'✅ Created config file at {path_to_data_yaml}')

    # Print file contents
    with open(path_to_data_yaml, 'r') as f:
        print("\nFile contents:\n")
        print(f.read())

# Example usage
dataset_root = "C:/Yolo/Fruit Yolo/custom_data"   # root dataset folder
path_to_classes_txt = os.path.join(dataset_root, "classes.txt")
path_to_data_yaml   = "C:/Yolo/Fruit Yolo/data.yaml"

create_data_yaml(path_to_classes_txt, path_to_data_yaml, dataset_root)
