import struct

class PnmImage(object):

    def __init__(self):
        self._filename = None
        self._image = []
        self._version = None
        self._maxval = None
        self._width = 0
        self._height = 0

    def reset(self):
        """Reset all image data (dims, data,...)"""
        self._filename = None
        self._image = []
        self._version = None
        self._maxval = None
        self._width = 0
        self._height = 0

    def _load_bin_pbm(self, fd):
        """Load image data from a file of PBM binary format

        :param fd: file descriptor of the file starting to the image data so
            the version and dimensions should have been skipped
        """
        c = fd.read(1)
        i = 0
        size = self._width * self._height
        while len(c) > 0:
            c = struct.unpack("<B", c)[0]
            # each byte is a bit array of 8 pixels
            for b in range(8):
                if c & 128:
                    self._image.append(1)
                else:
                    self._image.append(0)
                c <<= 1
                i += 1
                if i >= size:
                    break
            if i >= size:
                break
            c = fd.read(1)

    def _load_bin_pgm(self, fd):
        """Load image data from a file of PGM binary format

        :param fd: file descriptor of the file starting to the image data so
            the version and dimensions should have been skipped
        """
        maxval = fd.readline().decode("utf-8").strip()
        self._maxval = int(maxval)
        c = fd.read(1)
        while len(c) > 0:
            c = struct.unpack("<B", c)[0]
            self._image.append(c)
            c = fd.read(1)

    def load(self, filename):
        """Load an image pnm. Managed formats: P4, P5

        :param filename: the image's file name
        :return: False if failed to open the file

        """
        ret = True
        try:
            fd = open(filename, 'rb')
        except IOError:
            ret = False
        else:
            with fd:
                self.reset()
                self._filename = filename
                self._version = fd.readline().decode("utf-8").strip()
                c = fd.read(1).decode("utf-8")
                if c == "#":
                    comment = fd.readline().decode("utf-8").rstrip()
                    #print("#{}".format(comment))
                    c = ""
                dims = c + fd.readline().decode("utf-8").strip()
                width, height = dims.split(" ")
                self._width = int(width)
                self._height = int(height)
                if self._version == "P5":
                    self._load_bin_pgm(fd)
                elif self._version == "P4":
                    self._load_bin_pbm(fd)
        return ret

    def get_data(self):
        """Get the image data in a 1D list"""
        return self._image

    def get_data_bin(self, threshold=125):
        """Get the image data in a 1D list"""
        return [1 if e < threshold else 0 for e in self._image]

    def get_info(self):
        """Get the image information

        :return: information in a dict with keys: "version", "dims", "size", "max_value"
        """
        info = {
            "version": self._version,
            "dims": (self._width, self._height),
            "size": len(self._image),
            "max_value": self._maxval
        }
        return info

    def get_size(self):
        return len(self._image)

    def show_image_info(self):
        print("file: {}".format(self._filename))
        print("version: {}".format(self._version))
        print("dims: {}x{} (={})".format(self._width, self._height, self._width*self._height))
        print("image size: {}".format(len(self._image)))
        print("max val : {}".format(self._maxval))
        print("5 first pixels: {}".format(self._image[:5]))
        print("5 last pixels: {}".format(self._image[-5:]))


if __name__ == "__main__":
    img = PnmImage()
    img.load("data/ext_ln0_car0.pgm")
    img.show_image_info()
    print()
    img.load("mc_p22.pgm")
    img.show_image_info()
    print()
    img.load("tst.pbm")
    img.show_image_info()
