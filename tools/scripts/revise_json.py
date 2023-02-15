import os
import fire
import json5
import json

def revise_json(json_path, key, new_value):
    json_content = json5.load(open(json_path, 'r'))
    json_content[key] = new_value
    print(f"Update key {key} to {new_value}")
    json_name = os.path.basename(json_path).split('.')[0]
    new_json_name = json_name + '_revised' + '.json5'
    new_json_path = os.path.join(os.path.dirname(json_path), new_json_name)
    with open(new_json_path, 'w') as f:
        json.dump(json_content, f, indent = 4)


if __name__ == "__main__":
    fire.Fire()