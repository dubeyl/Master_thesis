import ruamel.yaml
config_path = "C:/Master/Master_thesis/automation/config.yaml"

yaml = ruamel.yaml.YAML()
# yaml.preserve_quotes = True
with open(config_path) as fp:
    data = yaml.load(fp)
# print(data)
data["folders_muscle"] = "'C:/Master/Master_thesis/automation/data/muscle'"
data["default_muscle_video_framerate"] = 20

with open(config_path, "w") as f:
    yaml.dump(data, f)

import yaml
config_path = "C:/Master/Master_thesis/automation/config.yaml"

with open(config_path) as f:
    list_doc = yaml.safe_load(f)

list_doc["folders_muscle"] = "'C:/Master/Master_thesis/automation/data/muscle'"
list_doc["default_muscle_video_framerate"] = 20

with open(config_path, "w") as f:
    yaml.dump(list_doc, f)