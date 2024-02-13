import yaml

from synthetic_data import config

def save_config(config, path):
    """
    traverse through all classes in the config module, 
    for each a class, create a instance of that class (the init does not need arguments),
    then save the class name, and its attributes to a yaml file at path
    """

    # create a dictionary to store the attributes of the config classes
    config_dict = {}

    # traverse through all classes in the config module
    for name, cls in config.__dict__.items():
        if isinstance(cls, type):
            # create a instance of the class
            instance = cls()

            # create a dictionary to store the attributes of the instance
            instance_dict = {}

            # traverse through all attributes of the instance
            for attr, value in instance.__dict__.items():
                # store the attribute and its value in the instance dictionary
                instance_dict[attr] = value

            # store the instance dictionary in the config dictionary
            config_dict[name] = instance_dict

    # save the config dictionary to a yaml file at path
    with open(path, 'w') as file:
        yaml.dump(config_dict, file)