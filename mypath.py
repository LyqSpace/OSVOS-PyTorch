from util.path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/Ship01/Dataset/DAVIS'

    @staticmethod
    def save_root_dir():
        return './models'

    @staticmethod
    def models_dir():
        return "./models"

    @staticmethod
    def db_flood_root_dir():
        return '/Ship01/Dataset/flood/realworld'
