from mmengine import Config
import toml
cfg = Config.fromfile(config)
with open(temp_dir/"output/a.toml", "w") as fp:
    toml.dump(cfg.to_dict(), fp)