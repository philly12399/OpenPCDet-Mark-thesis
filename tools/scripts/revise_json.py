import os
import fire
import json5
import json


def revise_json(json_path, key, new_value):
    json_content = json5.load(open(json_path, 'r'))
    json_content[key] = new_value
    print(f"Update key {key} to {new_value}")
    with open(json_path, 'w') as f:
        json.dump(json_content, f, indent=4)


if __name__ == "__main__":
    fire.Fire()
