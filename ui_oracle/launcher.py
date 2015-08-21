import Pyro4
import threading
import sys
from ui_oracle.transform_oracle import TransformService

__author__ = 'maeglin89273'


def register_services(daemon):
    name_uris = {}
    name_uris["oracle.transform"] = daemon.register(TransformService())

    return name_uris


def dispatch_new_thread_to_nameserver():
    thread = threading.Thread(target=Pyro4.naming.startNSloop)
    thread.setDaemon(True)
    thread.start()


def start_service_daemon():
    daemon = Pyro4.Daemon()
    name_uris = register_services(daemon)

    name_server = Pyro4.locateNS()

    for name, uri in name_uris.items():
        name_server.register(name, uri)
    print("server is up")
    daemon.requestLoop()


def main():
    sys.excepthook = Pyro4.util.excepthook
    # dispatch_new_thread_to_nameserver()
    start_service_daemon()


if __name__ == "__main__":
    main()
