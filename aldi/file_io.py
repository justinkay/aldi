from iopath.common.file_io import PathManager as PathManagerBase
from iopath.common.file_io import PathHandler

PathManager = PathManagerBase()
class ALDIHandler(PathHandler):
    """
    Resolve anything that's hosted under detectron2's namespace.
    """

    PREFIX = "models/"
    ALDI_WEIGHTS_PREFIX = "https://github.com/justinkay/aldi/releases/download/v0.0.1/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path, **kwargs):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.ALDI_WEIGHTS_PREFIX + name, **kwargs)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(
            self.ALDI_WEIGHTS_PREFIX + path[len(self.PREFIX) :], mode, **kwargs
        )
PathManager.register_handler(ALDIHandler)