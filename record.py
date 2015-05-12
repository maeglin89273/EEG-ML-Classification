import open_bci_v3 as bci

PORT = "/dev/ttyUSB0"

def handle_sample(sample):
  print(sample.channel_data)

board = bci.OpenBCIBoard(PORT)
board.print_register_settings()
board.start_streaming(handle_sample)