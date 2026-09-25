"""Entry point: builds the objects from Config and runs one configuration.

Batch size lives here and in Config only. No module takes it as a constructor
argument -- N is read from the tensor at runtime.
"""

from config import Config


def main():
    config = Config()
    raise NotImplementedError


if __name__ == "__main__":
    main()
