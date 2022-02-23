import os
import logging
import sys


class LoggingHandler:

    def __init__(self, debug_path, run_name):

        self.debug_path = debug_path
        self.run_name = run_name

    def setup_logging(self):

        # create directory if it doesn't exist
        if not os.path.isdir(self.debug_path):
            os.mkdir(self.debug_path)

        # generate filename
        debug_file = os.path.join(self.debug_path, 'debug.' + self.run_name + '.log')

        # delete older debug file if it exists
        if os.path.isfile(debug_file):
            os.remove(debug_file)

        # define basic logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        logger.handlers = []  # clear out previous handlers

        # write to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # write to debug file
        handler = logging.FileHandler(debug_file)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def draw_logo(self):
        """
        Write octopus logo to the log.
        Returns:None
        """

        logging.info('              _---_')
        logging.info('            /       \\')
        logging.info('           |         |')
        logging.info('   _--_    |         |    _--_')
        logging.info('  /__  \\   \\  0   0  /   /  __\\')
        logging.info('     \\  \\   \\       /   /  /')
        logging.info('      \\  -__-       -__-  /')
        logging.info('  |\\   \\    __     __    /   /|')
        logging.info('  | \\___----         ----___/ |')
        logging.info('  \\                           /')
        logging.info('   --___--/    / \\    \\--___--')
        logging.info('         /    /   \\    \\')
        logging.info('   --___-    /     \\    -___--')
        logging.info('   \\_    __-         -__    _/')
        logging.info('     ----               ----')
        logging.info('')
        logging.info('       O  C  T  O  P  U  S')
        logging.info('')
